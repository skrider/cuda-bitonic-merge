import math

def generate_schedule(n, block_width):
    schedule = []
    nblocks = n // block_width
    k = 2
    while k <= n:
        j = k // 2
        while j > 0:
            exceeds_width = False
            for bid in range(nblocks):
                iii = bid * block_width
                for ii in range(block_width):
                    i = ii + iii
                    l = i ^ j
                    if l >= iii + block_width or l < iii:
                        exceeds_width = True
                        break
            schedule.append((k, j, exceeds_width))
            j //= 2
        k *= 2
    return schedule

# max sram for an nvidia A10
max_sram = 49152
# for the case of 16-bit elements with no argsort
max_sram_elem = max_sram // 2
# for the case of 32-bit elements with longs for arg indices
min_sram_elem = max_sram // 10
max_sram_elem_log2 = int(math.log2(max_sram_elem) + 1)
min_sram_elem_log2 = int(math.log2(min_sram_elem))
log_block_widths = list(range(min_sram_elem_log2, max_sram_elem_log2 + 1))

log_ns = list(range(min_sram_elem_log2, 18))
ns = [2 ** i for i in log_ns]
block_widths = [2 ** i for i in log_block_widths]

code = ""
code += "#include <vector>\n"
code += "#include <stdexcept>\n"
code += "\n"
code += "static std::vector<std::vector<std::vector<int>>> schedules = {\n    "
members = []

for i in log_ns:
    for jj in log_block_widths:
        schedule = generate_schedule(2 ** i, 2 ** jj)
        acc = [[0, 0, 0, 0, False]] # k start, k end, j start, j end, gmem
        append_new = True

        for ii, s in enumerate(schedule):
            exceeds_width = s[2]
            if append_new or exceeds_width:
                k = s[0]
                j = s[1]
                acc[-1][1] = k
                acc[-1][3] = j
                acc.append([k, 0, j, 0, 1 if exceeds_width else 0])
                append_new = exceeds_width

        acc[-1][1] = schedule[-1][0]
        acc[-1][3] = schedule[-1][1]
        acc = acc[1:]

        members.append("{\n        " + ",\n        ".join([f"{{{x[0]}, {x[1]}, {x[2]}, {x[3]}, {int(x[4])}}}" for x in acc]) + "\n    }")

code += ",\n    ".join(members)
code += "\n};\n"

code += f"""
std::vector<std::vector<int>> get_schedule(int log_n, int log_block_width) {{
    int coord = (log_n - {log_ns[0]}) * {len(log_block_widths)} + (log_block_width - {log_block_widths[0]});
    if (coord < 0 || coord >= schedules.size()) {{
        throw std::runtime_error("Invalid kernel parameters");
    }}
    return schedules[coord];
}}
"""

print(code)
