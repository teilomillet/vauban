#!/usr/bin/env python3
"""
Verify the Vauban environment setup.
Checks Python version, dependencies, and API keys.
"""

import sys
import os
import importlib.util
from rich.console import Console
from rich.panel import Panel

console = Console()

def check_python_version():
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        console.print("[red]❌ Python 3.10+ is required.[/red]")
        return False
    console.print(f"[green]✅ Python {v.major}.{v.minor}.{v.micro}[/green]")
    return True

def check_import(module_name):
    if importlib.util.find_spec(module_name) is None:
        console.print(f"[red]❌ Module '{module_name}' not found.[/red]")
        return False
    console.print(f"[green]✅ Module '{module_name}' installed.[/green]")
    return True

def check_env_var(var_name, required=True):
    val = os.getenv(var_name)
    if not val:
        if required:
            console.print(f"[red]❌ Environment variable '{var_name}' is missing.[/red]")
            return False
        else:
            console.print(f"[yellow]⚠️  Environment variable '{var_name}' is missing (optional).[/yellow]")
            return True
    console.print(f"[green]✅ {var_name} is set.[/green]")
    return True

def main():
    console.print(Panel("[bold blue]Vauban Environment Verification[/bold blue]"))
    
    all_good = True
    
    console.print("\n[bold]System[/bold]")
    if not check_python_version():
        all_good = False
        
    console.print("\n[bold]Dependencies[/bold]")
    for mod in ["weave", "openai", "rich", "pydantic", "vauban"]:
        if not check_import(mod):
            all_good = False
            
    console.print("\n[bold]Configuration[/bold]")
    # Try loading .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
        console.print("[dim]Loaded .env file[/dim]")
    except ImportError:
        pass

    if not check_env_var("OPENAI_API_KEY", required=True):
        all_good = False
        
    check_env_var("WEAVE_PROJECT", required=False)
    
    if all_good:
        console.print("\n[bold green]🎉 System Ready! You can run 'vauban' now.[/bold green]")
        sys.exit(0)
    else:
        console.print("\n[bold red]💥 Issues found. Please fix them before running Vauban.[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
