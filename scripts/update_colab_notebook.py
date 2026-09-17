import json

nb_path = 'notebooks/Stream_ETHZ_Dataset_Colab.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        source_str = ''.join(source)
        if 'metadata_cols' in source_str:
            new_source = []
            skip = False
            for line in source:
                if 'metadata_cols = [' in line:
                    skip = True
                    continue
                if skip and ']' in line:
                    skip = False
                    continue
                if skip:
                    continue
                if 'meta_vec = np.array' in line:
                    new_source.append('        # Clean metadata only: total leaf blade area (exclude all 22 lesion segmentations)\n')
                    new_source.append('        la_tot = float(meta_dict.get("la_tot", 0.0) if not pd.isna(meta_dict.get("la_tot", 0.0)) else 0.0)\n')
                    continue
                if 'leaf_obs.append({"f": np.concatenate([vit_vec, meta_vec])' in line:
                    new_source.append('        leaf_obs.append({"f": np.concatenate([vit_vec, [la_tot]]), "d": placl, "l": lesion_area})\n')
                    continue
                if 'for start in range(len(leaf_obs) - SEQ_LEN + 1):' in line:
                    new_source.append('    # Strict Future Forecasting: Input [t-3, t-2, t-1, t] -> Target [t+1]\n')
                    new_source.append('    if len(leaf_obs) > SEQ_LEN:\n')
                    new_source.append('        for start in range(len(leaf_obs) - SEQ_LEN):\n')
                    continue
                if 'all_d_targets.append(win[-1]["d"])' in line:
                    new_source.append('            future_obs = leaf_obs[start + SEQ_LEN]\n')
                    new_source.append('            all_d_targets.append(future_obs["d"])\n')
                    continue
                if 'all_l_targets.append(win[-1]["l"])' in line:
                    new_source.append('            all_l_targets.append(future_obs["l"])\n')
                    continue
                new_source.append(line)
            cell['source'] = new_source

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Successfully updated Stream_ETHZ_Dataset_Colab.ipynb')
