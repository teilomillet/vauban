import asyncio
import pickle
import os
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
    CampaignEndEvent,
)

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

        self.console = Console()
        self.layout = self.make_layout()
        self.live = Live(self.layout, refresh_per_second=4, screen=True)
        
        # State for UI
        self.attacks: List[Dict] = []
        self.stats = {
            "generation": 0,
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
            Layout(name="stats", size=10),
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

        details = Table.grid(expand=True)
        details.add_column(ratio=1)
        details.add_row(line)
        details.add_row(
            f"[dim]Wave {self.stats['generation']}/{self.engine.max_generations} · Attacks {self.stats['attacks']} · Goal: {self.goal}[/dim]"
        )

        return Panel(details, title="[bold]Attack Vector[/bold]", border_style="cyan", padding=(0, 1))

    def generate_log(self) -> Panel:
        table = Table(box=box.SIMPLE, expand=True, show_header=False)
        table.add_column("Time", style="dim", width=8)
        table.add_column("Status", width=10)
        table.add_column("Score", width=6)
        table.add_column("Details")

        # Show last 10 attacks
        for attack in self.attacks[-10:]:
            time_str = attack["time"]
            score = attack["score"]
            is_breach = attack["is_breach"]
            prompt = attack["prompt"][:50] + "..."
            
            status_color = "red" if is_breach else "green"
            status_text = "BREACH" if is_breach else "DEFLECTED"
            
            table.add_row(
                time_str,
                f"[{status_color}]{status_text}[/{status_color}]",
                f"{score:.1f}",
                f"{prompt}"
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
        
        grid.add_row("Generation", f"[bold]{self.stats['generation']}/{self.engine.max_generations}[/bold]")
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

    def generate_footer(self) -> Panel:
        return Panel(
            "Press [bold]Ctrl+C[/bold] to pause/quit. Session will be saved automatically.",
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

    async def run(self):
        with Live(self.layout, refresh_per_second=4, screen=True) as live:
            try:
                async for event in self.engine.run_stream():
                    if isinstance(event, CampaignStartEvent):
                        pass
                    
                    elif isinstance(event, WaveStartEvent):
                        self.stats["generation"] = event.generation
                    
                    elif isinstance(event, AttackResultEvent):
                        self.attacks.append({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "score": event.score,
                            "is_breach": event.is_breach,
                            "prompt": event.attack.prompt
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
                        
                        # Update footer to prompt user
                        self.layout["footer"].update(Panel(
                            "Campaign Complete. Press [bold]Enter[/bold] to return to War Room.",
                            style="white on green",
                        ))
                        live.refresh()
                        
                        # Wait for user input to dismiss
                        await asyncio.to_thread(input)
                        
                        # Delete session file on successful completion
                        if os.path.exists(SESSION_FILE):
                            os.remove(SESSION_FILE)
                        return

                    self.update_ui()
                    
            except asyncio.CancelledError:
                pass
