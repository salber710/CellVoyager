import os
import shutil
from IPython.display import display, Markdown, Code, Image
import ipywidgets as widgets

class AnalysisDashboard:
    def __init__(self, outputs_dir="outputs", favorites_dir="favorites"):
        self.outputs_dir = outputs_dir
        self.favorites_dir = favorites_dir

        # Load all analysis folders
        self.analysis_folders = sorted([
            os.path.join(outputs_dir, f) for f in os.listdir(outputs_dir)
            if os.path.isdir(os.path.join(outputs_dir, f))
        ])

        self.dropdown = widgets.Dropdown(
            options=[(os.path.basename(f), f) for f in self.analysis_folders],
            description='Analysis:',
            layout=widgets.Layout(width='50%')
        )

        self.button = widgets.Button(description="Load Analysis", button_style='success')
        self.tag_button = widgets.Button(description="Tag as Interesting", button_style='info')
        self.tag_button.on_click(self.tag_analysis)

        display(self.dropdown, self.button, self.tag_button)

    def load_analysis(self, b):
        folder = self.dropdown.value
        display(Markdown(f"## 📂 {os.path.basename(folder)}"))

        # Display plan
        plan_path = os.path.join(folder, "plan.txt")
        if os.path.exists(plan_path):
            with open(plan_path, 'r') as f:
                plan_text = f.read()
            display(Markdown(f"### Hypothesis / Plan\n{plan_text}"))

        # Display code
        code_path = os.path.join(folder, "code.py")
        if os.path.exists(code_path):
            with open(code_path, 'r') as f:
                code_text = f.read()
            display(Markdown("### Python Code"))
            display(Code(code_text, language='python'))

        # Display reflection
        reflection_path = os.path.join(folder, "conclusion.txt")
        if os.path.exists(reflection_path):
            with open(reflection_path, 'r') as f:
                reflection_text = f.read()
            display(Markdown(f"### Reflection / Conclusion\n{reflection_text}"))

        # Display figures
        fig_files = sorted([f for f in os.listdir(folder) if f.startswith("figure_")])
        if fig_files:
            display(Markdown("### Figures"))
            for fig_file in fig_files:
                fig_path = os.path.join(folder, fig_file)
                display(Image(filename=fig_path))

    def tag_analysis(self, b):
        if not hasattr(self, 'selected_folder'):
            return
        folder = self.selected_folder
        new_favorite_folder = os.path.join(self.favorites_dir, os.path.basename(folder))
        shutil.copytree(folder, new_favorite_folder)
        self.analysis_folders.append(new_favorite_folder)
        self.dropdown.options = [(os.path.basename(f), f) for f in self.analysis_folders]
        self.dropdown.value = new_favorite_folder
        display(Markdown(f"Tagged analysis as interesting: {os.path.basename(folder)}"))

# To run it:
# dashboard = AnalysisDashboard(outputs_dir="outputs", favorites_dir="favorites")
# (make sure you're in a Jupyter environment!)
