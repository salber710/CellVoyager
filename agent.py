import openai
import os
import json
import scanpy as sc
import nbformat as nbf
import pandas as pd
from nbconvert.preprocessors import ExecutePreprocessor
import tempfile
import numpy as np
import gc
import datetime
from logger import Logger
import base64
import h5py

AVAILABLE_PACKAGES = "scanpy, scvi-tools, scVelo, CellTypist, anndata, matplotlib, numpy, seaborn, pandas, scipy"
class AnalysisAgent:
    def __init__(self, h5ad_path, paper_summary_path, openai_api_key, model_name, analysis_name, max_iterations=5, prompt_dir="prompts"):
        self.h5ad_path = h5ad_path
        self.paper_summary = open(paper_summary_path).read()
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.analysis_name = analysis_name
        self.max_iterations = max_iterations
        self.prompt_dir = prompt_dir
        
        self.completed_analyses = []
        self.failed_analyses = []
        self.output_dir = "outputs"
        self.client = openai.OpenAI(api_key=openai_api_key)
        
        # Initialize code memory to track the last few cells of code
        self.code_memory = []
        self.code_memory_size = 3  # Number of code cells to remember

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Coding guidelines: guide agent on how to write code and conduct analyses
        self.coding_guidelines = open(os.path.join(self.prompt_dir, "coding_guidelines.txt")).read()
        self.coding_guidelines = self.coding_guidelines.format(name=self.analysis_name, adata_path=self.h5ad_path, available_packages=AVAILABLE_PACKAGES)

        # System prompt for coding agents
        self.coding_system_prompt = open(os.path.join(self.prompt_dir, "coding_system_prompt.txt")).read()

        # Initialize logger: keeps track of all actions, prompts, responses, errors, etc.
        self.logger = Logger()
        self.logger.log_action(
            "Agent initialized", 
            f"h5ad_path: {h5ad_path}\n" +
            f"model: {model_name}\n" +
            f"max_iterations: {max_iterations}"
        )
        # Initialize notebook executor
        self.executor = ExecutePreprocessor(timeout=600, kernel_name='python3')

        # Load the .obs data from the anndata file
        print("Loading anndata .obs for summarization...")
        self.adata_obs = self.load_h5ad_obs(self.h5ad_path)
        self.adata_summary = self.summarize_adata_metadata()
        print("ADATA SUMMARY: ", self.adata_summary)
        self.logger.log_action("Data loaded and summarized", self.adata_summary)
        print(f"✅ Loaded {self.h5ad_path}")
        


    def summarize_adata_metadata(self, length_cutoff=10):
        """
        Summarize the agent's anndata metadata

        Args:
            length_cutoff (int): How many max unique values to include for each metadata column
        """
        summarization_str = f"Below is a description of the columns in adata.obs: \n"
        columns = self.adata_obs.columns
        for col in columns:
            unique_vals = np.unique(self.adata_obs[col])
            summarization_str += f"Column {col} contains the values {unique_vals[:length_cutoff]} \n"
        return summarization_str

    def load_h5ad_obs(self, h5ad_path):
        """Load just the .obs data from an h5ad file while preserving data types"""
        with h5py.File(h5ad_path, 'r') as f:
            obs_dict = {}
            
            # Process each column in obs
            for k in [k for k in f['obs'].keys() if not k.startswith('_')]:
                data = f['obs'][k][:]
                
                # Handle categorical data
                if 'categories' in f['obs'][k].attrs:
                    try:
                        # Get category values (handling references if needed)
                        cat_ref = f['obs'][k].attrs['categories']
                        if isinstance(cat_ref, h5py.h5r.Reference):
                            # Dereference to get categories
                            categories = [str(x) for x in f[cat_ref][:]]
                        else:
                            # Normal categories
                            cat_vals = f['obs'][k].attrs['categories'][:]
                            categories = cat_vals.asstr()[:] if hasattr(cat_vals, 'asstr') else [str(v) for v in cat_vals]
                        
                        # Create categorical data
                        data = pd.Categorical.from_codes(
                            data.astype(int) if not np.issubdtype(data.dtype, np.integer) else data,
                            categories=categories
                        )
                    except Exception as e:
                        print(f"Warning: Error with categorical {k}: {str(e)}")
                        data = np.array([str(x) for x in data])
                
                # Handle string data
                elif data.dtype.kind in ['S', 'O'] or h5py.check_string_dtype(f['obs'][k].dtype) is not None:
                    try:
                        if data.dtype.kind == 'S':
                            data = np.array([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in data])
                        elif hasattr(f['obs'][k], 'asstr'):
                            data = f['obs'][k].asstr()[:]
                    except Exception:
                        pass
                
                obs_dict[k] = data
            
            # Get index
            try:
                if '_index' in f['obs']:
                    idx = f['obs']['_index']
                    index = idx.asstr()[:] if hasattr(idx, 'asstr') else np.array([str(x) for x in idx[:]])
                else:
                    index = None
            except Exception:
                index = None
        
        # Create dataframe
        df = pd.DataFrame(obs_dict, index=index)
        print(f"Loaded obs data: {len(df)} rows × {len(df.columns)} columns")
        return df

    def generate_initial_analysis(self, attempted_analyses):
        prompt = open(os.path.join(self.prompt_dir, "first_draft.txt")).read()
        prompt = prompt.format(CODING_GUIDELINES=self.coding_guidelines, adata_summary=self.adata_summary, 
                               past_analyses=attempted_analyses, paper_txt=self.paper_summary)

        
        self.logger.log_prompt("user", prompt, "Initial Analysis")
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        result = response.choices[0].message.content
        
        analysis = json.loads(result)
        return analysis
    
    def update_code_memory(self, notebook_cells):
        """Update the code memory with the latest code cells from the notebook"""
        # Extract code cells from the notebook
        code_cells = []
        for cell in notebook_cells:
            if cell.get('cell_type') == 'code':
                code_cells.append(cell['source'])
                
        # Keep only the most recent cells up to code_memory_size
        self.code_memory = code_cells[-self.code_memory_size:] if len(code_cells) > 0 else []
        
    def generate_next_step_analysis(self, analysis, attempted_analyses, notebook_cells, results_interpretation):
        hypothesis = analysis["hypothesis"]
        analysis_plan = analysis["analysis_plan"]
        first_step_code = analysis["first_step_code"]
        
        # Update code memory with latest notebook cells
        self.update_code_memory(notebook_cells)
        
        # Use the code memory for generating the next step
        recent_code = "\n\n# Next Cell\n".join(reversed(self.code_memory))

        prompt = open(os.path.join(self.prompt_dir, "next_step.txt")).read()
        prompt = prompt.format(hypothesis=hypothesis, analysis_plan=analysis_plan, first_step_code=first_step_code,
                               CODING_GUIDELINES=self.coding_guidelines, results_interpretation=results_interpretation,
                               previous_code=recent_code, adata_summary=self.adata_summary, past_analyses=attempted_analyses,
                               paper_txt=self.paper_summary)
        
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        result = response.choices[0].message.content

        analysis = json.loads(result)
        return analysis

    def critique_step(self, analysis, past_analyses, notebook_cells):
        hypothesis = analysis["hypothesis"]
        analysis_plan = analysis["analysis_plan"]
        first_step_code = analysis["first_step_code"]

        if notebook_cells is None:
            recent_code = ""
        else:
            # Update code memory with latest notebook cells
            self.update_code_memory(notebook_cells)
            
            # Use the code memory for generating the next step
            recent_code = "\n\n# Next Cell\n".join(reversed(self.code_memory))

        prompt = open(os.path.join(self.prompt_dir, "critic.txt")).read()
        prompt = prompt.format(hypothesis=hypothesis, analysis_plan=analysis_plan, first_step_code=first_step_code,
                               CODING_GUIDELINES=self.coding_guidelines, adata_summary=self.adata_summary, past_analyses=past_analyses,
                               paper_txt=self.paper_summary, previous_code=recent_code)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a single-cell bioinformatics expert providing feedback on code and analysis plan."},
                {"role": "user", "content": prompt}
            ]
        )
        feedback = response.choices[0].message.content
        return feedback

    def incorporate_critique(self, analysis, feedback, notebook_cells):
        ## Return analysis object
        hypothesis = analysis["hypothesis"]
        analysis_plan = analysis["analysis_plan"]
        first_step_code = analysis["first_step_code"]

        if notebook_cells is None:
            recent_code = ""
        else:
            # Update code memory with latest notebook cells
            self.update_code_memory(notebook_cells)

        recent_code = "\n\n# Next Cell\n".join(reversed(self.code_memory))

        prompt = open(os.path.join(self.prompt_dir, "incorporate_critque.txt")).read()
        prompt = prompt.format(hypothesis=hypothesis, analysis_plan=analysis_plan, first_step_code=first_step_code,
                               CODING_GUIDELINES=self.coding_guidelines, adata_summary=self.adata_summary,
                               feedback=feedback, previous_code=recent_code)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        modified_analysis = json.loads(response.choices[0].message.content)

        self.logger.log_prompt("user", prompt, "Incorporate Critiques")

        return modified_analysis
    
    def fix_code(self, code, error):
        """Attempts to fix code that produced an error"""
        prompt = f"""Fix this code that produced an error:
        
        Code:
        ```python
        {code}
        ```
        
        Error:
        {error}
        
        Provide only the fixed code with no explanation.
        You can only use the following packages: {AVAILABLE_PACKAGES}
        """
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a coding assistant helping to fix code."},
                {"role": "user", "content": prompt}
            ]
        )
        fixed_code = response.choices[0].message.content
        
        return fixed_code

    def interpret_results(self, notebook, past_analyses):
        # Get the last cell
        last_cell = notebook.cells[-1]
        no_interpretation = "No results found"

        if last_cell.get('cell_type') != 'code':
            print("Last cell is not a code cell")
            return no_interpretation
        
        #### Extract text output ####
        text_output = ""
        if 'outputs' in last_cell:
            for output in last_cell['outputs']:
                if output.get('output_type') == 'stream': # print statements
                    text_output += output.get('text', '')
                elif output.get('output_type') == 'execute_result': # variable outputs e.g. df.head()
                    text_output += str(output.get('data', {}).get('text/plain', ''))
        
        #### Extract image outputs ####
        image_outputs = []
        if 'outputs' in last_cell:
            for output in last_cell['outputs']:
                if output.get('output_type') == 'display_data':
                    image_data = output.get('data', {}).get('image/png')
                    if image_data:
                        image_outputs.append({
                            'data': image_data,
                            'format': 'image/png'
                        })

        if not text_output and not image_outputs: # no output found
            return no_interpretation
        
        user_content = []
        prompt = open(os.path.join(self.prompt_dir, "interp_results.txt")).read()
        prompt = prompt.format(text_output=text_output, paper_txt=self.paper_summary,
                               CODING_GUIDELINES=self.coding_guidelines, past_analyses=past_analyses)
        user_content.append({"type": "text", "text": prompt})

        for img in image_outputs:
            try:
                # Get the image data 
                image_data = img['data']
                
                # Remove the base64 prefix if present
                if isinstance(image_data, str) and "," in image_data:
                    image_data = image_data.split(",")[1]
                
                # Add the image to the content
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }
                })
            except Exception as e:
                print(f"Warning: Error processing image: {str(e)}")
                
        response = self.client.chat.completions.create(
            model = "gpt-4o",
            messages = [
                {"role": "system", "content": "You are a single-cell bioinformatics expert providing feedback on code and analysis plan."},
                {"role": "user", "content": user_content}
            ]
        )
        feedback = response.choices[0].message.content
        
        self.logger.log_prompt("user", user_content, "Results Interpretation")
        
        return feedback
    
    def get_feedback(self, analysis, past_analyses, notebook_cells, iterations=1):
        current_analysis = analysis
        for i in range(iterations):
            feedback = self.critique_step(current_analysis, past_analyses, notebook_cells)
            current_analysis = self.incorporate_critique(current_analysis, feedback, notebook_cells)

        return current_analysis


    def run(self, num_analyses=1, max_fix_attempts=3):
        past_analyses = ""

        for analysis_idx in range(num_analyses):  # Try 5 analyses
            # Reset code memory for this analysis
            self.code_memory = []
            
            print(f"\n🚀 Starting Analysis {analysis_idx+1}")

            analysis = self.generate_initial_analysis(past_analyses)
            hypothesis = analysis["hypothesis"]                
            analysis_plan = analysis["analysis_plan"]
            initial_code = analysis["first_step_code"]
            
            # Log only the output of the analysis
            self.logger.log_response(f"Hypothesis: {hypothesis}\n\nAnalysis Plan:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(analysis_plan)]) + f"\n\nInitial Code:\n{initial_code}", "initial_analysis")
            
            # Get feedback on initial analysis plan
            modified_analysis = self.get_feedback(analysis, past_analyses, None)

            hypothesis = modified_analysis["hypothesis"]                
            analysis_plan = modified_analysis["analysis_plan"]
            current_code = modified_analysis["first_step_code"]
            
            # Log revised analysis plan
            self.logger.log_response(f"Revised Hypothesis: {hypothesis}\n\nRevised Analysis Plan:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(analysis_plan)]) + f"\n\nRevised Code:\n{current_code}", "revised_analysis")
            
            # Create a markdown cell with the analysis plan
            plan_markdown = "# Analysis Plan\n\n**Hypothesis**: " + hypothesis + "\n\n## Steps:\n"
            for i, step in enumerate(analysis_plan):
                plan_markdown += f"{i+1}. {step}\n"

            # Create initial notebook with the hypothesis and plan
            notebook = self.create_initial_notebook(hypothesis)
            
            # Add the analysis plan as a markdown cell
            notebook.cells.append(nbf.v4.new_markdown_cell(plan_markdown))
            
            # Add the first analysis code cell
            notebook.cells.append(nbf.v4.new_code_cell(initial_code))

            for iteration in range(self.max_iterations):
                # Execute the notebook
                success, error_msg, notebook = self.execute_notebook(notebook)

                if success:
                    results_interpretation = self.interpret_results(notebook, past_analyses)
                    # Log the interpretation
                    self.logger.log_response(results_interpretation, "results_interpretation")

                else:
                    print(f"⚠️ Code errored with: {error_msg}")
                    fix_attempt, fix_successful = 0, False
                    while fix_attempt < max_fix_attempts and not fix_successful:
                        fix_attempt += 1
                        print(f"  🔧 Fix attempt {fix_attempt}/{max_fix_attempts}")

                        current_code = self.fix_code(current_code, error_msg)
                        notebook.cells[-1] = nbf.v4.new_code_cell(current_code)

                        success, error_msg, notebook = self.execute_notebook(notebook)

                        if success:
                            fix_successful = True
                            print(f"  ✅ Fix successful on attempt {fix_attempt}")
                            break
                        else:
                            print(f"  ❌ Fix attempt {fix_attempt} failed")

                            if fix_attempt == max_fix_attempts:
                                print(f"  ⚠️ Failed to fix after {max_fix_attempts} attempts. Moving to next iteration.")
                                results_interpretation = "Current analysis step failed to run. Try an alternative approach"
                    results_interpretation = self.interpret_results(notebook, past_analyses)

                analysis = {"hypothesis": hypothesis, "analysis_plan": analysis_plan, "first_step_code": current_code}
                next_step_analysis = self.generate_next_step_analysis(analysis, past_analyses, notebook.cells, results_interpretation)

                # Get feedback on the next step(s)
                print("Getting feedback on the next step(s)")
                modified_analysis = self.get_feedback(next_step_analysis, past_analyses, notebook.cells)
                
                # Log the next step
                self.logger.log_response(f"Next step: {modified_analysis['analysis_plan'][0]}\n\nCode:\n```python\n{modified_analysis['first_step_code']}\n```", "next_step")

                # Add the next step to the notebook
                analysis_step_plan = modified_analysis["analysis_plan"][0]
                notebook.cells.append(nbf.v4.new_markdown_cell(f"## {analysis_step_plan}"))
                notebook.cells.append(nbf.v4.new_code_cell(modified_analysis["first_step_code"]))
                
                # Update the code memory with the new cell
                self.update_code_memory(notebook.cells)

            # Save the notebook
            notebook_path = os.path.join(self.output_dir, f"analysis_{analysis_idx+1}.ipynb")
            with open(notebook_path, 'w', encoding='utf-8') as f:
                nbf.write(notebook, f)
            print(f"💾 Saved notebook to: {notebook_path}")

            # TODO: modify this to include the entire analysis plan
            past_analyses += f"{hypothesis}\n"

    def create_initial_notebook(self, hypothesis):
        notebook = nbf.v4.new_notebook()
        
        # Add markdown cell with hypothesis
        notebook.cells.append(nbf.v4.new_markdown_cell(f"# Analysis\n\n**Hypothesis**: {hypothesis}"))
        
        # Add setup code to import libraries and load data with enhanced visualization setup
        setup_code = f"""import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Set up visualization defaults for better plots
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.figsize = (8, 8)
sc.settings.dpi = 100
sc.settings.facecolor = 'white'
warnings.filterwarnings('ignore')

# Set Matplotlib and Seaborn styles for better visualization
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['savefig.dpi'] = 150
sns.set_style('whitegrid')
sns.set_context('notebook', font_scale=1.2)

# Load data
print("Loading data...")
adata = sc.read_h5ad("{self.h5ad_path}")
print(f"Data loaded: {{adata.shape[0]}} cells and {{adata.shape[1]}} genes")
"""
        notebook.cells.append(nbf.v4.new_code_cell(setup_code))
        
        self.logger.log_action("Created initial notebook", f"Setup code:\n```python\n{setup_code}\n```")
        return notebook

    def execute_notebook(self, notebook):
        """Execute only the last cell of the notebook while preserving previous cell outputs"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save realtime copy for user viewing
                realtime_dir = os.path.join(self.output_dir, "realtime")
                os.makedirs(realtime_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                realtime_path = os.path.join(realtime_dir, f"notebook_{timestamp}.ipynb")
                
                # Save pre-execution notebook
                with open(realtime_path, 'w', encoding='utf-8') as f:
                    nbf.write(notebook, f)
                    print(f"💾 Saved notebook to: {realtime_path}")
                
                # Create temporary execution notebook
                temp_path = os.path.join(temp_dir, "temp_notebook.ipynb")
                with open(temp_path, 'w', encoding='utf-8') as f:
                    nbf.write(notebook, f)
                
                # Read notebook for execution
                with open(temp_path, encoding='utf-8') as f:
                    nb = nbf.read(f, as_version=4)
                
                # Create notebook with just the last cell for execution
                last_cell_nb = nbf.v4.new_notebook()
                # Copy all cells except last (with outputs preserved)
                for i in range(len(nb.cells) - 1):
                    last_cell_nb.cells.append(nb.cells[i])
                # Add the last cell to execute
                last_cell_nb.cells.append(nb.cells[-1])
                
                # Execute only the last cell
                self.executor.preprocess(last_cell_nb, {'metadata': {'path': temp_dir}})
                
                # Update original notebook with executed cell
                nb.cells[-1] = last_cell_nb.cells[-1]
                
                # Save executed notebook
                with open(temp_path, 'w', encoding='utf-8') as f:
                    nbf.write(nb, f)
                with open(temp_path, encoding='utf-8') as f:
                    executed_nb = nbf.read(f, as_version=4)
                with open(realtime_path, 'w', encoding='utf-8') as f:
                    nbf.write(executed_nb, f)
                    print(f"💾 Saved executed notebook to: {realtime_path}")
                
                # Check for errors in the last cell
                last_cell = executed_nb.cells[-1]
                if last_cell.cell_type == 'code' and hasattr(last_cell, 'outputs'):
                    for output in last_cell.outputs:
                        if output.output_type == 'error':
                            error_name = output.ename
                            error_value = output.evalue
                            traceback = "\n".join(output.traceback) if hasattr(output, 'traceback') else ""
                            error_msg = self.logger.format_traceback(error_name, error_value, traceback)
                            self.logger.log_error(error_msg, last_cell['source'])
                            return False, error_msg, executed_nb
                
                # Update code memory with the successfully executed notebook
                self.update_code_memory(executed_nb.cells)
                
                return True, None, executed_nb
                
        except Exception as e:
            print(f"⚠️ Notebook execution failed: {str(e)}")
            return False, str(e), notebook


    