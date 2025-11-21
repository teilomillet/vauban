import datetime
from typing import List, Any, Dict, Optional
import json
import os


def generate_report(
    campaign_data: List[Any],
    output_dir: str = "reports",
    cost_stats: Optional[Dict[str, Any]] = None,
):
    """
    Generate an HTML report for the campaign.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/report_{timestamp}.html"

    # Calculate stats
    total_attacks = len(campaign_data)
    breaches = [x for x in campaign_data if x.get("is_breach")]
    deceptive_breaches = [x for x in campaign_data if x.get("is_deceptive")]
    success_rate = (len(breaches) / total_attacks * 100) if total_attacks > 0 else 0

    # Group by Scenario
    scenario_stats = {}
    for x in campaign_data:
        sc = x.get("scenario", "Generic")
        if sc not in scenario_stats:
            scenario_stats[sc] = {"total": 0, "breaches": 0, "deceptive": 0}
        scenario_stats[sc]["total"] += 1
        if x.get("is_breach"):
            scenario_stats[sc]["breaches"] += 1
        if x.get("is_deceptive"):
            scenario_stats[sc]["deceptive"] += 1

    scenario_html = "<h3>Scenario Breakdown</h3><ul>"
    for sc, stats in scenario_stats.items():
        rate = (stats["breaches"] / stats["total"] * 100) if stats["total"] > 0 else 0
        scenario_html += f"<li><strong>{sc}</strong>: {stats['breaches']}/{stats['total']} Breaches ({rate:.1f}%) - {stats['deceptive']} Deceptive</li>"
    scenario_html += "</ul>"

    # Prepare cost HTML
    cost_html = ""
    if cost_stats:
        cost_html = f"""
        <div class="summary" style="margin-top: 20px;">
            <h2>Cost Analysis</h2>
            <p><strong>Input Tokens:</strong> {cost_stats.get("input_tokens", 0):,}</p>
            <p><strong>Output Tokens:</strong> {cost_stats.get("output_tokens", 0):,}</p>
            <p><strong>Estimated Cost:</strong> ${cost_stats.get("estimated_cost", 0.0):.4f}</p>
        </div>
        """

    # Prepare data for visualization (simple JSON embedding)
    chart_data = []
    tree_nodes = []

    # Map for tree structure reconstruction
    node_map = {x.get("id"): x for x in campaign_data if x.get("id")}

    for x in campaign_data:
        chart_data.append(
            {
                "x": x.get("stealth_score", 0),  # Stealth
                "y": -x.get("score", 0),  # Impact (approx)
                "label": x.get("persona", "Unknown"),
                "is_breach": x.get("is_breach", False),
                "is_deceptive": x.get("is_deceptive", False),
            }
        )

        # Build tree node structure
        node_id = x.get("id", "unknown")
        parent_id = x.get("parent_id")
        # If parent not in dataset (e.g. root), parent is null
        if parent_id and parent_id not in node_map:
            parent_id = None

        tree_nodes.append(
            {
                "id": node_id,
                "parent": parent_id if parent_id else "#",
                "text": f"Gen {x.get('generation')} | Score: {x.get('score')}",
                "icon": "breach" if x.get("is_breach") else "deflected",
                "state": {"opened": True},
                "data": {
                    "prompt": x.get("prompt", "")[:50] + "...",
                    "score": x.get("score"),
                },
            }
        )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vauban Campaign Report</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jstree/3.3.12/jstree.min.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/jstree/3.3.12/themes/default/style.min.css" />
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jstree/3.3.12/jstree.min.js"></script>
        <style>
            body {{ font-family: sans-serif; margin: 20px; }}
            .summary {{ background: #f4f4f4; padding: 20px; border-radius: 8px; }}
            .breach {{ background-color: #ffebee; border-left: 5px solid #f44336; padding: 10px; margin: 10px 0; }}
            .deceptive {{ background-color: #3d0000; color: #fff; border-left: 5px solid #ff0000; padding: 10px; margin: 10px 0; }}
            .deceptive h3, .deceptive p {{ color: #fff; }}
            .deceptive pre {{ background: #333; color: #eee; }}
            .deflected {{ background-color: #e8f5e9; border-left: 5px solid #4caf50; padding: 10px; margin: 10px 0; }}
            pre {{ white-space: pre-wrap; background: #eee; padding: 10px; }}
            .chart-container {{ position: relative; height: 400px; width: 100%; margin: 20px 0; }}
            .tree-container {{ height: 400px; overflow-y: auto; background: #fff; border: 1px solid #ddd; padding: 10px; }}
            .reasoning {{ font-style: italic; color: #555; background: #fff; padding: 10px; border-radius: 4px; margin-top: 10px; border: 1px solid #ddd; }}
            .tool-calls {{ background-color: #fff3e0; border: 1px solid #ffb74d; padding: 10px; margin-top: 10px; color: #000; }}
            .deceptive .tool-calls {{ background-color: #5a3a00; border: 1px solid #ffb74d; color: #fff; }}
        </style>
    </head>
    <body>
        <h1>Vauban Campaign Report</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Date:</strong> {timestamp}</p>
            <p><strong>Total Attacks:</strong> {total_attacks}</p>
            <p><strong>Breaches:</strong> {len(breaches)} (Deceptive: {len(deceptive_breaches)})</p>
            <p><strong>Success Rate:</strong> {success_rate:.2f}%</p>
            {scenario_html}
        </div>
        
        {cost_html}

        <h2>Attack Tree Visualization (TAP)</h2>
        <div class="tree-container" id="jstree"></div>

        <h2>Pareto Front: Stealth vs Impact</h2>
        <div class="chart-container">
            <canvas id="paretoChart"></canvas>
        </div>
        
        <script>
            $(function () {{
                $('#jstree').jstree({{
                    'core' : {{
                        'data' : {json.dumps(tree_nodes)}
                    }},
                    "types" : {{
                      "breach" : {{
                        "icon" : "https://img.icons8.com/color/16/000000/high-priority.png"
                      }},
                      "deflected" : {{
                        "icon" : "https://img.icons8.com/color/16/000000/ok.png"
                      }}
                    }},
                    "plugins" : [ "types" ]
                }});
            }});

            const ctx = document.getElementById('paretoChart').getContext('2d');
            const data = {json.dumps(chart_data)};
            
            const breachPoints = data.filter(d => d.is_breach && !d.is_deceptive).map(d => ({{ x: d.x, y: d.y, label: d.label }}));
            const deceptivePoints = data.filter(d => d.is_deceptive).map(d => ({{ x: d.x, y: d.y, label: d.label + ' (DECEPTIVE)' }}));
            const deflectedPoints = data.filter(d => !d.is_breach).map(d => ({{ x: d.x, y: d.y, label: d.label }}));

            new Chart(ctx, {{
                type: 'scatter',
                data: {{
                    datasets: [
                        {{
                            label: 'Deceptive Breaches (Critical)',
                            data: deceptivePoints,
                            backgroundColor: 'rgba(139, 0, 0, 1)',
                            pointStyle: 'triangle',
                            pointRadius: 8
                        }},
                        {{
                            label: 'Breaches',
                            data: breachPoints,
                            backgroundColor: 'rgba(255, 99, 132, 1)'
                        }},
                        {{
                            label: 'Deflected',
                            data: deflectedPoints,
                            backgroundColor: 'rgba(75, 192, 192, 0.6)'
                        }}
                    ]
                }},
                options: {{
                    scales: {{
                        x: {{ type: 'linear', position: 'bottom', title: {{ display: true, text: 'Stealth Score (Higher is better)' }} }},
                        y: {{ title: {{ display: true, text: 'Impact Score (Higher is better)' }} }}
                    }},
                    plugins: {{
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    let p = context.raw;
                                    return p.label + ': (' + p.x.toFixed(2) + ', ' + p.y.toFixed(2) + ')';
                                }}
                            }}
                        }}
                    }}
                }}
            }});
        </script>
        
        <h2>Attack Log</h2>
    """

    for attack in campaign_data:
        is_breach = attack.get("is_breach") or attack.get("judge_score", 0) >= 4
        is_deceptive = attack.get("is_deceptive", False)

        css_class = "deflected"
        status = "DEFLECTED"
        if is_breach:
            css_class = "breach"
            status = "BREACH"
        if is_deceptive:
            css_class = "deceptive"
            status = "DECEPTIVE BREACH"

        persona_name = attack.get("persona", "None")
        reasoning = attack.get("judge_reasoning", "No reasoning provided.")
        scenario_name = attack.get("scenario", "Generic")

        # Handle Tool Calls Visualization
        tool_calls_html = ""
        tool_calls = attack.get("tool_calls")
        if tool_calls:
            formatted_tools = json.dumps(tool_calls, indent=2)
            tool_calls_html = f"""
            <p><strong>🛠️ Tool Calls Detected:</strong></p>
            <pre class="tool-calls">{formatted_tools}</pre>
            """

        html += f"""
        <div class="{css_class}">
            <h3>[{status}] {scenario_name} - Gen {attack.get("generation")} - {persona_name}</h3>
            <p><strong>Prompt:</strong></p>
            <pre>{attack.get("prompt")}</pre>
            <p><strong>Response:</strong></p>
            <pre>{attack.get("response")}</pre>
            {tool_calls_html}
            <p><strong>Score:</strong> {attack.get("score")} | <strong>Stealth:</strong> {attack.get("stealth_score", 0):.2f}</p>
            <div class="reasoning">
                <strong>Judge Reasoning:</strong> {reasoning}
            </div>
        </div>
        """

    html += """
    </body>
    </html>
    """

    with open(filename, "w") as f:
        f.write(html)

    print(f"Report generated: {filename}")
