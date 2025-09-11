#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <iomanip>
#include <algorithm>
#include <sqlite3.h>

// 从 feature_processor.h 复制必要的常量
const std::string DB_PATH = "/mnt/nfs/perimeter_data.db.load";
constexpr int EMB_FACE_DIM = 512;
constexpr int EMB_BODY_DIM = 2048;


void write_state_to_file(const std::string &filepath,
                         int next_gid,
                         const std::map<std::string, std::vector<std::vector<float>>> &faces,
                         const std::map<std::string, std::vector<std::vector<float>>> &bodies) {
    std::ofstream out(filepath);
    out << "Next GID: " << next_gid << "\n\n";

    std::set<std::string> all_gids;
    for (const auto &pair: faces) all_gids.insert(pair.first);
    for (const auto &pair: bodies) all_gids.insert(pair.first);

    for (const auto &gid: all_gids) {
        out << "--- GID: " << gid << " ---\n";

        if (faces.count(gid)) {
            const auto &face_feats = faces.at(gid);
            out << "Faces: " << face_feats.size() << "\n";
            for (size_t i = 0; i < face_feats.size(); ++i) {
                out << "  Face[" << i << "]: ";
                for (float val: face_feats[i]) {
                    out << std::fixed << std::setprecision(8) << val << " ";
                }
                out << "\n";
            }
        }

        if (bodies.count(gid)) {
            const auto &body_feats = bodies.at(gid);
            out << "Bodies: " << body_feats.size() << "\n";
            for (size_t i = 0; i < body_feats.size(); ++i) {
                out << "  Body[" << i << "]: ";
                for (float val: body_feats[i]) {
                    out << std::fixed << std::setprecision(8) << val << " ";
                }
                out << "\n";
            }
        }
        out << "\n";
    }
    out.close();
    std::cout << "State successfully written to " << filepath << std::endl;
}

int main() {
    sqlite3 *db;
    if (sqlite3_open_v2(DB_PATH.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        std::cerr << "Error: Can't open database: " << sqlite3_errmsg(db) << std::endl;
        return 1;
    }
    std::cout << "Database opened successfully for reading." << std::endl;

    std::map<std::string, std::vector<std::vector<float>>> loaded_faces;
    std::map<std::string, std::vector<std::vector<float>>> loaded_bodies;
    int max_gid_num = 0;

    const char *sql = "SELECT gid, type, feature FROM prototypes ORDER BY gid, type, idx;";
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "Error: Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return 1;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        std::string gid = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0));
        std::string type = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));
        const float *feature_blob = static_cast<const float *>(sqlite3_column_blob(stmt, 2));
        int bytes = sqlite3_column_bytes(stmt, 2);

        if (feature_blob) {
            int num_floats = bytes / sizeof(float);
            std::vector<float> feature(feature_blob, feature_blob + num_floats);

            if (type == "faces" && num_floats == EMB_FACE_DIM) {
                loaded_faces[gid].push_back(feature);
            } else if (type == "bodies" && num_floats == EMB_BODY_DIM) {
                loaded_bodies[gid].push_back(feature);
            }
        }

        try {
            if (gid.rfind("G", 0) == 0) {
                max_gid_num = std::max(max_gid_num, std::stoi(gid.substr(1)));
            }
        } catch (const std::exception &) {
            // Ignore parse errors
        }
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);

    int next_gid = max_gid_num + 1;

    std::cout << "Loaded " << loaded_faces.size() << " GIDs with faces and "
              << loaded_bodies.size() << " GIDs with bodies from DB." << std::endl;
    std::cout << "Calculated Next GID: " << next_gid << std::endl;

    write_state_to_file("/mnt/nfs/state_after_reload.txt", next_gid, loaded_faces, loaded_bodies);

    return 0;
}