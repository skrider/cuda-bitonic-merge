#include <vector>
#include <stdexcept>

static std::vector<std::vector<int>> schedules = {
    {},
    {},
    {},
    {},
    {78},
    {},
    {},
    {},
    {78, 91, 92},
    {91},
    {},
    {},
    {78, 91, 92, 105, 106, 107},
    {91, 105, 106},
    {105},
    {},
    {78, 91, 92, 105, 106, 107, 120, 121, 122, 123},
    {91, 105, 106, 120, 121, 122},
    {105, 120, 121},
    {120},
    {78, 91, 92, 105, 106, 107, 120, 121, 122, 123, 136, 137, 138, 139, 140},
    {91, 105, 106, 120, 121, 122, 136, 137, 138, 139},
    {105, 120, 121, 136, 137, 138},
    {120, 136, 137}
};

std::vector<int> get_schedule(int log_n, int log_block_width) {
    int coord = (log_n - 12) * 4 + (log_block_width - 12);
    if (coord < 0 || coord >= schedules.size()) {
        throw std::runtime_error("Invalid kernel parameters");
    }
    return schedules[coord];
}
