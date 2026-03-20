import sys
import os
from typing import List

# Ensure we can import from the core and ga directories
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from InquirerPy import inquirer
from InquirerPy.base.control import Choice

from core.mix import Mix
from core.audio_module import CompressorModule
from ga.population import Population

class GenMixCLI:
    def __init__(self):
        self.console = Console()
        # Initialize with 5 mixes
        initial_mixes = [Mix(crossovers=[150.0, 2500.0]) for _ in range(5)]
        # Seed them with some initial FX
        for m in initial_mixes:
            m.bands[0].modules.append(CompressorModule())
            m.bands[1].modules.append(CompressorModule())
            
        self.population = Population(initial_mixes)
        self.structural_rate = 0.5
        self.parametric_rate = 0.5
        self.parent_index = 0

    def display_population(self):
        self.console.clear()
        self.console.print(Panel.fit(
            f"[bold cyan]GenMix Evolution System[/bold cyan]\n"
            f"Generation: [yellow]{self.population.generation_count}[/yellow] | "
            f"Parent Index: [green]{self.parent_index}[/green]\n"
            f"Structural Rate: [magenta]{self.structural_rate}[/magenta] | "
            f"Parametric Rate: [magenta]{self.parametric_rate}[/magenta]",
            box=box.DOUBLE_EDGE, border_style="blue"
        ))

        table = Table(title="Current Population", box=box.ROUNDED, expand=True)
        table.add_column("ID", justify="center", style="cyan", no_wrap=True)
        table.add_column("Low Band", style="green")
        table.add_column("Mid Band", style="yellow")
        table.add_column("High Band", style="red")

        for i, mix in enumerate(self.population.mixes):
            row_style = "bold white on grey15" if i == self.parent_index else ""
            
            # Format band info
            band_info = []
            for band in mix.bands:
                if not band.modules:
                    band_info.append("[grey50]Empty[/grey50]")
                else:
                    chain = " -> ".join([m.name for m in band.modules])
                    band_info.append(chain)
            
            table.add_row(
                str(i),
                band_info[0],
                band_info[1],
                band_info[2],
                style=row_style
            )

        self.console.print(table)

    def select_parent(self):
        choices = [Choice(i, name=f"Mix {i}") for i in range(len(self.population.mixes))]
        self.parent_index = inquirer.select(
            message="Select parent for next generation:",
            choices=choices,
            default=self.parent_index
        ).execute()

    def set_rates(self):
        self.structural_rate = float(inquirer.text(
            message="Set structural mutation rate (0.0 - 1.0):",
            default=str(self.structural_rate),
            validate=lambda x: 0.0 <= float(x) <= 1.0
        ).execute())
        self.parametric_rate = float(inquirer.text(
            message="Set parametric mutation rate (0.0 - 1.0):",
            default=str(self.parametric_rate),
            validate=lambda x: 0.0 <= float(x) <= 1.0
        ).execute())

    def lock_parameters_menu(self):
        # Nested menu to lock/unlock parameters of the selected parent
        parent = self.population.mixes[self.parent_index]
        
        band_choices = [Choice(i, name=f"Band: {band.name}") for i, band in enumerate(parent.bands)]
        band_choices.append(Choice(None, name="[Back]"))
        
        band_idx = inquirer.select(message="Select band:", choices=band_choices).execute()
        if band_idx is None: return

        modules = parent.bands[band_idx].modules
        if not modules:
            self.console.print("[red]No modules in this band.[/red]")
            return

        mod_choices = [Choice(i, name=f"Module: {m.name}") for i, m in enumerate(modules)]
        mod_choices.append(Choice(None, name="[Back]"))
        
        mod_idx = inquirer.select(message="Select module:", choices=mod_choices).execute()
        if mod_idx is None: return

        params = modules[mod_idx].parameters
        param_choices = []
        for p_name, p in params.items():
            status = "LOCKED" if p.is_locked else "Unlocked"
            param_choices.append(Choice(p_name, name=f"{p_name} ({status})"))
        
        param_choices.append(Choice(None, name="[Back]"))
        
        p_name = inquirer.select(message="Toggle lock on parameter:", choices=param_choices).execute()
        if p_name:
            params[p_name].is_locked = not params[p_name].is_locked
            self.console.print(f"[green]Toggled {p_name}.[/green]")

    def run(self):
        while True:
            self.display_population()
            action = inquirer.select(
                message="Main Menu:",
                choices=[
                    "Evolve Next Gen",
                    "Select Parent",
                    "Lock Parameter",
                    "Set Mutation Rates",
                    "Exit"
                ]
            ).execute()

            if action == "Exit":
                break
            elif action == "Select Parent":
                self.select_parent()
            elif action == "Set Mutation Rates":
                self.set_rates()
            elif action == "Evolve Next Gen":
                self.population.generate_next_generation(
                    self.parent_index, 
                    self.structural_rate, 
                    self.parametric_rate
                )
                self.console.print("[bold green]Evolved![/bold green]")
            elif action == "Lock Parameter":
                self.lock_parameters_menu()

if __name__ == "__main__":
    cli = GenMixCLI()
    cli.run()
