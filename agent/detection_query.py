from __future__ import annotations


def _format_dataframe_blocks(title: str, dfs) -> str:
    if not dfs:
        return ""
    blocks = []
    for idx, df in enumerate(dfs):
        if df is None:
            continue
        blocks.append(f"##### {title} {idx}\n{df.to_string(index=False, header=False)}\n")
    return "".join(blocks)


def build_detection_agent_v3_query(
    curr_dfs,
    mode: str,
    last_rule: str | None = None,
    additional_dfs=None,
    reference_dfs=None,
    anomaly_types: str | None = None,
) -> str:
    final_query = _format_dataframe_blocks("DATA", curr_dfs)

    if additional_dfs is not None:
        if mode == "train-combined-fn":
            final_query += _format_dataframe_blocks("NORMAL DATA", additional_dfs)
        elif mode == "train-combined-fp":
            final_query += _format_dataframe_blocks("ABNORMAL DATA", additional_dfs)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    if reference_dfs is not None:
        final_query += _format_dataframe_blocks("NORMAL REFERENCE", reference_dfs)

    if anomaly_types is not None:
        anomaly_types_str = "\n".join(f"#  {split}" for split in anomaly_types.split("\n"))
        final_query += (
            "##### Anomaly Types BEGIN #####\n"
            + anomaly_types_str
            + "\n##### Anomaly Types END #####\n"
        )

    if last_rule:
        final_query += "##### CODE FROM LAST ITERATION\n" + last_rule

    return final_query
