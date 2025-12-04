import asyncio
import pickle
import os
from statistics import mean
from typing import Optional, List, Any, Dict
from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich import box
from rich.panel import Panel
from rich.table import Table

from vauban.engine import SiegeEngine
from vauban.interfaces import (
    CampaignStartEvent,
    WaveStartEvent,
    AttackResultEvent,
    WaveSummaryEvent,
    WaveSummaryEvent,
    CampaignEndEvent,
)
from vauban.formatting import AttackFormatter

LOGO = """
╦  ╦╔═╗╦ ╦╔╗ ╔═╗╔╗╔
╚╗╔╝╠═╣║ ║╠╩╗╠═╣║║║
 ╚╝ ╩ ╩╚═╝╚═╝╩ ╩╝╚╝
"""

SESSION_FILE = ".vauban_session.pkl"

def save_session(engine: SiegeEngine, filename: str = SESSION_FILE):
    """Save the current engine state to a file."""
    try:
        with open(filename, "wb") as f:
            pickle.dump(engine, f)
        print(f"\n[green]Session saved to {filename}[/green]")
    except Exception as e:
        print(f"\n[red]Failed to save session: {e}[/red]")

def load_session(filename: str = SESSION_FILE) -> Optional[SiegeEngine]:
    """Load a saved engine state."""
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, "rb") as f:
            engine = pickle.load(f)
        print(f"[green]Resumed session from {filename}[/green]")
        return engine
    except Exception as e:
        print(f"[red]Failed to load session: {e}[/red]")
        return None

class VaubanCLI:
    def __init__(
        self,
        engine: SiegeEngine,
        target_model: str,
        goal: str,
        history: Optional[List[Any]] = None,
    ):
        # Keep references for UI labels and telemetry
        self.engine = engine
        self.target_model = target_model
        self.goal = goal
        self.history = history or []
        self.restart_requested = False
        self.command_request: Optional[str] = None

        self.console = Console()
        self.layout = self.make_layout()
        # Live screen is constructed per run() to keep setup lightweight
        self.live = None

        # State for UI
        self.attacks: List[Dict] = []
        self.stats = {
            "generation": 0,  # current wave index
            "total_waves": engine.max_generations,
            "waves_remaining": engine.max_generations,
            "best_score": 0.0,
            "survivors": 0,
            "breaches": 0,
            "cost": 0.0,
            "attacks": 0,
            "successes": 0,
        }

        # Animation state for attack vector lane
        self._vector_len = 22
        self._vector_step = 0
        self._stop_event: Optional[asyncio.Event] = None
        self._ticker_task: Optional[asyncio.Task] = None
        self._input_task: Optional[asyncio.Task] = None

    def make_layout(self) -> Layout:
        layout = Layout()
        
        # Split into Header, Body, Footer
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        
        # Split Body into History (Left) and Main (Right)
        layout["body"].split_row(
            Layout(name="history", size=30),
            Layout(name="main"),
        )

        # Split Main into Attack Vector, Stats, Log
        layout["main"].split_column(
            Layout(name="vector", size=5),
            Layout(name="stats", size=8),
            Layout(name="log"),
        )
        
        return layout

    def generate_header(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_row(f"[bold cyan]{LOGO}[/bold cyan]")
        grid.add_row(f"[dim]Target Goal: {self.goal}[/dim]")
        return Panel(grid, style="white on black")

    def generate_attack_vector(self) -> Panel:
        """Show a single point traversing the lane to depict live traffic."""
        bar = ["-"] * self._vector_len
        pos = self._vector_step % self._vector_len
        bar[pos] = "[yellow]o[/yellow]"
        lane = "".join(bar)

        source = "[green]Vauban[/green]"
        destination = f"[magenta]{self.target_model}[/magenta]"
        line = f"{source} |{lane}| {destination}"

        total_waves = self.stats.get("total_waves", self.engine.max_generations)
        remaining = max(total_waves - self.stats["generation"], 0)
        details = Table.grid(expand=True)
        details.add_column(ratio=1)
        details.add_row(line)
        details.add_row(
            f"[dim]Wave {self.stats['generation']}/{total_waves} (left {remaining}) · Attacks {self.stats['attacks']} · Goal: {self.goal}[/dim]"
        )

        return Panel(details, title="[bold]Attack Vector[/bold]", border_style="cyan", padding=(0, 1))

    def generate_log(self) -> Panel:
        table = Table(box=box.ROUNDED, expand=True, show_header=True, header_style="bold white")
        table.add_column("Time", style="dim", width=8)
        table.add_column("Wave", width=6, justify="right")
        table.add_column("Status", width=12)
        table.add_column("Score", width=6, justify="right")
        table.add_column("Details")

        # Show last 20 attacks (newest on top)
        for attack in reversed(self.attacks[-20:]):
            time_str = attack["time"]
            wave = attack.get("wave", "-")
            score = attack["score"]
            is_breach = attack["is_breach"]
            prompt = attack["prompt"]
            strategy = attack.get("strategy", "Unknown")
            tool_calls = attack.get("tool_calls", [])
            metadata = attack.get("metadata", {})
            
            # Format details
            prompt_display = prompt[:60] + "..." if len(prompt) > 60 else prompt
            tools_str = AttackFormatter.format_tools(tool_calls)
            meta_str = AttackFormatter.format_metadata(metadata)
            
            details = f"[{strategy}] {prompt_display}{tools_str}{meta_str}"
            
            label, color = AttackFormatter.get_status_label(is_breach)
            
            table.add_row(
                time_str,
                str(wave),
                f"[{color}]{label}[/{color}]",
                f"{score:.1f}",
                details
            )
            
        return Panel(
            table, 
            title="[bold]Live Attack Feed[/bold]", 
            border_style="blue",
            padding=(0, 1)
        )

    def generate_stats(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="left")
        grid.add_column(justify="right")

        total_waves = self.stats.get("total_waves", self.engine.max_generations)
        remaining = max(total_waves - self.stats["generation"], 0)
        
        grid.add_row("Generation", f"[bold]{self.stats['generation']}/{total_waves}[/bold]")
        grid.add_row("Waves Left", f"{remaining}")
        grid.add_row("Best Score", f"[bold yellow]{self.stats['best_score']:.2f}[/bold yellow]")
        grid.add_row("Survivors", f"{self.stats['survivors']}")
        grid.add_row("Total Breaches", f"[red]{self.stats['breaches']}[/red]")
        grid.add_row("Est. Cost", f"${self.stats['cost']:.4f}")
        
        return Panel(
            grid, 
            title="[bold]Campaign Intel[/bold]", 
            border_style="green",
            padding=(1, 2)
        )

    def _sparkline(self, scores: List[float]) -> str:
        """
        ASCII sparkline mapping scores (0-5) into . - = * # for quick trend scan.
        """
        levels = [".", "-", "=", "*", "#", "#"]
        return "".join(levels[min(5, max(0, int(round(s))))] for s in scores) or "-"

    def _render_campaign_summary(self, result):
        history = result.history if hasattr(result, "history") else []
        scores = [r.get("score", 0) for r in history]
        breaches = sum(1 for r in history if r.get("is_breach"))
        best = max(scores) if scores else 0.0
        last = scores[-1] if scores else 0.0
        avg = mean(scores) if scores else 0.0
        goal_hit = best >= 4.0
        reasoning = history[-1].get("judge_reasoning", "") if history else ""
        trend = self._sparkline(scores)

        lines = [
            f"Goal achieved: {'YES' if goal_hit else 'NO'} (best={best:.1f}, last={last:.1f}, avg={avg:.1f})",
            f"Breaches: {breaches} / {len(history)}",
            f"Trend: {trend}",
        ]
        
        if result.termination_reason:
            lines.insert(0, f"[bold red]STOPPED: {result.termination_reason}[/bold red]")
            
        if reasoning:
            lines.append(f"Last reasoning: {reasoning[:180]}{'...' if len(reasoning) > 180 else ''}")
        return "\n".join(lines)

    def generate_footer(self) -> Panel:
        return Panel(
            "Running... | Type 'again', 'exit', or 'attack ...' and press Enter | Ctrl+C to force quit",
            style="white on blue",
        )

    def generate_history(self) -> Panel:
        table = Table(show_header=True, header_style="bold magenta", box=None, expand=True)
        table.add_column("Model", style="cyan")
        table.add_column("Score", justify="right")
        
        for entry in self.history:
            # Assuming entry is a SiegeResult object
            model = getattr(entry, "attacker_model", "Unknown")
            # If SiegeResult doesn't store model, we might need to pass it differently or rely on what's there
            # For now, let's assume we can get it or it's passed in the entry tuple
            if isinstance(entry, dict):
                 model = entry.get("model", "Unknown")
                 score = f"{entry.get('score', 0.0):.1f}"
            else:
                 # Fallback
                 score = "?"
            
            table.add_row(model, score)
            
        return Panel(
            table,
            title="[bold]Campaign History[/bold]",
            border_style="blue",
        )

    def update_ui(self):
        self.layout["header"].update(self.generate_header())
        self.layout["history"].update(self.generate_history())
        self.layout["vector"].update(self.generate_attack_vector())
        self.layout["stats"].update(self.generate_stats())
        self.layout["log"].update(self.generate_log())
        self.layout["footer"].update(self.generate_footer())

        # Advance animation frame so the lane moves each refresh
        self._vector_step = (self._vector_step + 1) % self._vector_len

    async def _ticker(self, stop_event: asyncio.Event):
        """Periodic refresh so the attack lane moves even when no events arrive."""
        try:
            while not stop_event.is_set():
                self.update_ui()
                await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            pass

    async def _watch_commands(self, stop_event: asyncio.Event, live: Live):
        """Listen for user commands; uses live console so prompt renders on the alt-screen."""
        # TODO: Replace blocking stdin with a non-blocking input widget so keystrokes echo inside Live without flicker.
        try:
            while not stop_event.is_set():
                # Use live.console.input so the prompt is drawn even while Live owns the screen.
                # We use an empty prompt to avoid fighting with the footer layout.
                cmd = await asyncio.to_thread(lambda: live.console.input(""))
                cmd = cmd.strip()
                if not cmd:
                    # Stay in the UI; ignore empty lines
                    continue

                lowered = cmd.lower()
                if lowered in {":q", "quit", "exit"}:
                    self.command_request = "__exit__"
                    stop_event.set()
                    break

                if lowered == "again":
                    self.restart_requested = True
                    stop_event.set()
                    break

                # Any other text is treated as an inline command for a new attack
                self.command_request = cmd
                stop_event.set()
                break
        except asyncio.CancelledError:
            pass

    async def run(self):
        self._stop_event = asyncio.Event()
        with Live(self.layout, refresh_per_second=4, screen=True) as live:
            # Start background ticker and command listener
            self._ticker_task = asyncio.create_task(self._ticker(self._stop_event))
            self._input_task = asyncio.create_task(self._watch_commands(self._stop_event, live))

            try:
                async for event in self.engine.run_stream():
                    if isinstance(event, CampaignStartEvent):
                        # Sync configured total waves so remaining math is accurate even if engine overrides
                        self.stats["total_waves"] = getattr(event, "max_generations", self.engine.max_generations)
                        self.stats["waves_remaining"] = self.stats["total_waves"]
                    
                    elif isinstance(event, WaveStartEvent):
                        self.stats["generation"] = event.generation
                        self.stats["waves_remaining"] = max(self.stats["total_waves"] - event.generation, 0)
                    
                    elif isinstance(event, AttackResultEvent):
                        wave_idx = getattr(event, "generation", event.metadata.get("wave", 0))
                        self.attacks.append({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "wave": wave_idx,
                            "score": event.score,
                            "is_breach": event.is_breach,
                            "prompt": event.attack.prompt,
                            "strategy": getattr(event.attack, "strategy", "Unknown"),
                            "tool_calls": event.tool_calls,
                            "metadata": event.metadata,
                            "judge_reasoning": event.judge_reasoning
                        })
                        self.stats["attacks"] += 1
                        if event.is_breach:
                            self.stats["breaches"] += 1
                        
                        # Update cost if available
                        if hasattr(self.engine.strategy, "get_cost_estimate"):
                            self.stats["cost"] = self.engine.strategy.get_cost_estimate()

                    elif isinstance(event, WaveSummaryEvent):
                        self.stats["survivors"] = event.survivors
                        self.stats["best_score"] = max(self.stats["best_score"], event.best_score)

                    elif isinstance(event, CampaignEndEvent):
                        live.console.print("\n[bold green]Campaign Complete![/bold green]")
                        live.console.print(f"Result: {event.result}")
                        summary = self._render_campaign_summary(event.result)
                        live.console.print(Panel(summary, title="Outcome Summary", border_style="green", padding=(1, 2)))
                        
                        # Update footer to prompt user; offer rerun to stay in UI and launch again
                        self.layout["footer"].update(Panel(
                            "Campaign Complete. Type ':q' to exit, 'again' to rerun, or enter 'attack <model> [siege|scout] [gens] [squad] [goal=...]' to launch a new one.",
                            style="white on green",
                        ))
                        live.refresh()
                        # Wait here until the command watcher captures user intent
                        while not self._stop_event.is_set():
                            await asyncio.sleep(0.1)
                        # Delete session file on successful completion
                        if os.path.exists(SESSION_FILE):
                            os.remove(SESSION_FILE)
                        return event.result

                    if self._stop_event.is_set():
                        # User requested command/restart; bail out to outer loop
                        return None
                
                # If the stream ends unexpectedly, fall through to wait for user command/enter
                while not self._stop_event.is_set():
                    await asyncio.sleep(0.1)
                return None
            except asyncio.CancelledError:
                pass
            finally:
                self._stop_event.set()
                for task in (self._ticker_task, self._input_task):
                    if task:
                        task.cancel()
