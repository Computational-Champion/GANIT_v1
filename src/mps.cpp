// Free-format MPS reader written from scratch.
// Supports: NAME, OBJSENSE, ROWS, COLUMNS (incl. integer MARKERs), RHS,
// RANGES, BOUNDS (UP LO FX FR MI PL BV LI UI), ENDATA.
// Names must not contain spaces (true for Netlib, MIPLIB, Mittelmann sets).
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "ganit/lp.hpp"

namespace ganit {
namespace {

std::string slurp(const std::string& path) {
    bool gz = path.size() > 3 && path.compare(path.size() - 3, 3, ".gz") == 0;
    if (gz) {
#if defined(_WIN32)
        throw std::runtime_error("gzip input not supported on Windows; decompress first");
#else
        std::string cmd = "gzip -dc '" + path + "'";
        FILE* f = popen(cmd.c_str(), "r");
        if (!f) throw std::runtime_error("cannot run gzip on " + path);
        std::string out;
        char buf[1 << 16];
        size_t k;
        while ((k = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, k);
        pclose(f);
        if (out.empty()) throw std::runtime_error("empty or unreadable file " + path);
        return out;
#endif
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void split(const std::string& line, std::vector<std::string>& tok) {
    tok.clear();
    size_t i = 0, n = line.size();
    while (i < n) {
        while (i < n && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
        if (i >= n) break;
        size_t j = i;
        while (j < n && !std::isspace(static_cast<unsigned char>(line[j]))) ++j;
        tok.emplace_back(line, i, j - i);
        i = j;
    }
}

double num(const std::string& s, int lineno) {
    char* end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str()) throw std::runtime_error("MPS line " + std::to_string(lineno) +
                                                   ": bad number '" + s + "'");
    return v;
}

enum class Sec { None, Name, ObjSense, Rows, Columns, Rhs, Ranges, Bounds, QuadObj, QMatrix, End };

}  // namespace

LP read_mps(const std::string& path) {
    const std::string text = slurp(path);
    LP lp;
    std::unordered_map<std::string, int> row_of, col_of;
    std::unordered_set<std::string> ignored_n_rows;
    std::string obj_name;
    std::vector<char> row_type;  // 'E','L','G'
    std::vector<double> rhs;
    std::vector<int> tr, tc;
    std::vector<double> tv;
    std::vector<char> lower_set;
    std::vector<int> qr, qc;
    std::vector<double> qv;
    bool int_mode = false;
    int neg_up_warnings = 0;

    Sec sec = Sec::None;
    std::vector<std::string> tok;
    std::istringstream in(text);
    std::string line;
    int lineno = 0;
    std::string last_col;
    int cur_col = -1;

    auto get_col = [&](const std::string& name) -> int {
        auto it = col_of.find(name);
        if (it == col_of.end())
            throw std::runtime_error("MPS: unknown column '" + name + "' at line " +
                                     std::to_string(lineno));
        return it->second;
    };

    while (std::getline(in, line)) {
        ++lineno;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '*') continue;
        split(line, tok);
        if (tok.empty()) continue;

        if (!std::isspace(static_cast<unsigned char>(line[0]))) {  // section header
            const std::string& h = tok[0];
            if (h == "NAME") {
                sec = Sec::Name;
                if (tok.size() > 1) lp.name = tok[1];
            } else if (h == "OBJSENSE") {
                sec = Sec::ObjSense;
                if (tok.size() > 1) lp.maximize = (tok[1] == "MAX" || tok[1] == "MAXIMIZE");
            } else if (h == "ROWS") sec = Sec::Rows;
            else if (h == "COLUMNS") sec = Sec::Columns;
            else if (h == "RHS") sec = Sec::Rhs;
            else if (h == "RANGES") sec = Sec::Ranges;
            else if (h == "BOUNDS") sec = Sec::Bounds;
            else if (h == "QUADOBJ") sec = Sec::QuadObj;
            else if (h == "QMATRIX" || h == "QSECTION") sec = Sec::QMatrix;
            else if (h == "ENDATA") { sec = Sec::End; break; }
            else if (h == "OBJSENSE" || h == "OBJSENCE") sec = Sec::ObjSense;
            else throw std::runtime_error("MPS: unknown section '" + h + "' at line " +
                                          std::to_string(lineno));
            continue;
        }

        switch (sec) {
            case Sec::ObjSense:
                lp.maximize = (tok[0] == "MAX" || tok[0] == "MAXIMIZE");
                break;
            case Sec::Rows: {
                if (tok.size() < 2) throw std::runtime_error("MPS: bad ROWS line " + std::to_string(lineno));
                char t = static_cast<char>(std::toupper(tok[0][0]));
                if (t == 'N') {
                    if (obj_name.empty()) obj_name = tok[1];
                    else ignored_n_rows.insert(tok[1]);
                } else {
                    row_of[tok[1]] = lp.m++;
                    lp.row_names.push_back(tok[1]);
                    row_type.push_back(t);
                    rhs.push_back(0.0);
                }
                break;
            }
            case Sec::Columns: {
                if (tok.size() >= 3 && tok[1] == "'MARKER'") {
                    if (tok[2] == "'INTORG'") int_mode = true;
                    else if (tok[2] == "'INTEND'") int_mode = false;
                    break;
                }
                if (tok[0] != last_col) {
                    last_col = tok[0];
                    cur_col = lp.n++;
                    col_of[tok[0]] = cur_col;
                    lp.col_names.push_back(tok[0]);
                    lp.c.push_back(0.0);
                    lp.l.push_back(0.0);
                    lp.u.push_back(kInf);
                    lp.is_int.push_back(int_mode ? 1 : 0);
                    lower_set.push_back(0);
                }
                for (size_t k = 1; k + 1 < tok.size(); k += 2) {
                    double v = num(tok[k + 1], lineno);
                    if (tok[k] == obj_name) { lp.c[cur_col] += v; continue; }
                    auto it = row_of.find(tok[k]);
                    if (it == row_of.end()) {
                        if (ignored_n_rows.count(tok[k])) continue;
                        throw std::runtime_error("MPS: unknown row '" + tok[k] + "' at line " + std::to_string(lineno));
                    }
                    tr.push_back(it->second);
                    tc.push_back(cur_col);
                    tv.push_back(v);
                }
                break;
            }
            case Sec::Rhs:
            case Sec::Ranges: {
                size_t start = (tok.size() % 2 == 1) ? 1 : 0;  // optional set name
                for (size_t k = start; k + 1 < tok.size(); k += 2) {
                    double v = num(tok[k + 1], lineno);
                    if (tok[k] == obj_name) {
                        if (sec == Sec::Rhs) lp.obj_const = -v;
                        continue;
                    }
                    auto it = row_of.find(tok[k]);
                    if (it == row_of.end()) {
                        if (ignored_n_rows.count(tok[k])) continue;
                        throw std::runtime_error("MPS: unknown row '" + tok[k] + "' at line " + std::to_string(lineno));
                    }
                    if (sec == Sec::Rhs) rhs[it->second] = v;
                    else {
                        // Store range temporarily in lo/hi after rows are finalised.
                        if (lp.lo.empty()) { lp.lo.assign(lp.m, std::nan("")); }
                        lp.lo[it->second] = v;
                    }
                }
                break;
            }
            case Sec::Bounds: {
                const std::string t = tok[0];
                bool needs_value = (t == "UP" || t == "LO" || t == "FX" || t == "LI" || t == "UI");
                std::string cname;
                double v = 0.0;
                if (needs_value) {
                    if (tok.size() >= 4) { cname = tok[2]; v = num(tok[3], lineno); }
                    else if (tok.size() == 3) { cname = tok[1]; v = num(tok[2], lineno); }
                    else throw std::runtime_error("MPS: bad BOUNDS line " + std::to_string(lineno));
                } else {
                    cname = tok.size() >= 3 ? tok[2] : tok[1];
                }
                int j = get_col(cname);
                if (t == "UP" || t == "UI") {
                    lp.u[j] = v;
                    if (v < 0 && lp.l[j] == 0.0 && !lower_set[j]) {
                        lp.l[j] = -kInf;
                        ++neg_up_warnings;
                    }
                    if (t == "UI") lp.is_int[j] = 1;
                } else if (t == "LO" || t == "LI") {
                    lp.l[j] = v; lower_set[j] = 1;
                    if (t == "LI") lp.is_int[j] = 1;
                } else if (t == "FX") { lp.l[j] = v; lp.u[j] = v; lower_set[j] = 1; }
                else if (t == "FR") { lp.l[j] = -kInf; lp.u[j] = kInf; lower_set[j] = 1; }
                else if (t == "MI") { lp.l[j] = -kInf; lower_set[j] = 1; }
                else if (t == "PL") { lp.u[j] = kInf; }
                else if (t == "BV") { lp.l[j] = 0; lp.u[j] = 1; lp.is_int[j] = 1; lower_set[j] = 1; }
                else throw std::runtime_error("MPS: unknown bound type '" + t + "'");
                break;
            }
            case Sec::QuadObj:
            case Sec::QMatrix: {
                if (tok.size() < 3) throw std::runtime_error("MPS: bad quadratic line " + std::to_string(lineno));
                int a = get_col(tok[0]), b = get_col(tok[1]);
                double v = num(tok[2], lineno);
                qr.push_back(a); qc.push_back(b); qv.push_back(v);
                if (sec == Sec::QuadObj && a != b) { qr.push_back(b); qc.push_back(a); qv.push_back(v); }
                break;
            }
            default:
                break;
        }
    }

    // Finalise row bounds from type, rhs and ranges.
    std::vector<double> range = lp.lo;  // NaN = no range
    lp.lo.assign(lp.m, 0.0);
    lp.hi.assign(lp.m, 0.0);
    for (int i = 0; i < lp.m; ++i) {
        double b = rhs[i];
        switch (row_type[i]) {
            case 'E': lp.lo[i] = b; lp.hi[i] = b; break;
            case 'L': lp.lo[i] = -kInf; lp.hi[i] = b; break;
            case 'G': lp.lo[i] = b; lp.hi[i] = kInf; break;
            default: throw std::runtime_error("MPS: unknown row type for " + lp.row_names[i]);
        }
        if (!range.empty() && !std::isnan(range[i])) {
            double r = range[i];
            if (row_type[i] == 'E') { if (r > 0) lp.hi[i] = b + r; else lp.lo[i] = b + r; }
            else if (row_type[i] == 'L') lp.lo[i] = b - std::fabs(r);
            else lp.hi[i] = b + std::fabs(r);
        }
    }
    lp.A = csr_from_triplets(lp.m, lp.n, tr, tc, tv);
    if (!qv.empty()) lp.Q = csr_from_triplets(lp.n, lp.n, qr, qc, qv);
    if (neg_up_warnings)
        std::cerr << "[mps] warning: " << neg_up_warnings
                  << " negative UP bounds with zero lower bound; lower set to -inf\n";
    return lp;
}

}  // namespace ganit
