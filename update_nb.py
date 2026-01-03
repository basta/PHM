import json

file_path = "/home/basta/Projects/PHM/time_explore.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

found = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        # Check if this is the target cell
        if any("def apply_sg(series, win=10, poly=3):" in line for line in source):
            # We found the cell.
            # We construct the new source
            new_source = []
            for line in source:
                # Use strict matching or substring matching that identifies the specific line
                if (
                    'plt.plot(apply_sg(sample.filter(like="Sensed_T").mean(axis=1)))'
                    in line
                ):
                    # Keep the newline if present in original, although json list usually has \n at end of strings
                    # except maybe the last one.
                    # source lines usually end with \n
                    if line.endswith("\n"):
                        new_source.append(
                            'plt.plot(sample.filter(like="Sensed_T").apply(apply_sg, axis=0))\n'
                        )
                    else:
                        new_source.append(
                            'plt.plot(sample.filter(like="Sensed_T").apply(apply_sg, axis=0))'
                        )
                else:
                    new_source.append(line)

            cell["source"] = new_source
            found = True
            break

if found:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully.")
else:
    print("Target cell not found.")
