import nbformat as nbf
import json
import openai
import os
import base64
from datetime import datetime
import matplotlib.pyplot as plt
import io
from PIL import Image
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import tempfile

class NotebookReporter:
    def __init__(self, openai_api_key, model_name="gpt-4-vision-preview"):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model_name = model_name
        
    def extract_notebook_content(self, notebook_path):
        """Extract content from the notebook including text, code, and figures"""
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbf.read(f, as_version=4)
        
        content = {
            'hypothesis': '',
            'analysis_plan': [],
            'figures': [],
            'findings': []
        }
        
        # Extract hypothesis and analysis plan from markdown cells
        for cell in notebook.cells:
            if cell.cell_type == 'markdown':
                text = cell.source
                if '**Hypothesis**:' in text:
                    content['hypothesis'] = text.split('**Hypothesis**:')[1].split('\n')[0].strip()
                elif '## Steps:' in text:
                    steps = text.split('## Steps:')[1].strip().split('\n')
                    content['analysis_plan'] = [step.strip('- ').strip() for step in steps if step.strip().startswith('-')]
            
            # Extract figures from code cell outputs
            elif cell.cell_type == 'code' and 'outputs' in cell:
                for output in cell.outputs:
                    if output.get('output_type') == 'display_data':
                        if 'image/png' in output.get('data', {}):
                            content['figures'].append({
                                'data': output['data']['image/png'],
                                'context': self._get_figure_context(cell)
                            })
        
        return content
    
    def _get_figure_context(self, cell):
        """Extract context around a figure from the cell"""
        context = {
            'code': cell.source,
            'text_output': ''
        }
        
        # Get any text output before the figure
        for output in cell.outputs:
            if output.get('output_type') == 'stream':
                context['text_output'] += output.get('text', '')
            elif output.get('output_type') == 'execute_result':
                context['text_output'] += str(output.get('data', {}).get('text/plain', ''))
        
        return context
    
    def generate_report(self, notebook_path, output_dir="reports"):
        """Generate a comprehensive report from the notebook content"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract content from notebook
        content = self.extract_notebook_content(notebook_path)
        
        # Prepare the prompt for the LLM
        prompt = self._create_report_prompt(content)
        
        # Generate the report using the LLM
        report_content = self._generate_report_with_llm(prompt, content['figures'])
        
        # Parse the report content to separate text and figure selections
        report_sections = self._parse_report_content(report_content)
        
        # Create PDF report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(output_dir, f"report_{timestamp}.pdf")
        self._create_pdf_report(pdf_path, report_sections, content['figures'])
        
        print(f"Report saved to: {pdf_path}")
        return pdf_path
    
    def _create_report_prompt(self, content):
        """Create a prompt for the LLM to generate the report"""
        prompt = f"""Please analyze this scientific analysis and generate a comprehensive report focusing on the most biologically significant findings. The report should include:

1. Core Hypothesis: {content['hypothesis']}

2. Analysis Procedure:
{chr(10).join([f"- {step}" for step in content['analysis_plan']])}

3. Key Findings:
Please analyze the figures and their context to identify the most significant biological findings. For each selected figure:
- Explain why this figure is biologically significant
- Describe the key patterns or relationships shown
- Interpret the biological implications
- Connect the findings to the original hypothesis

Please write a report that:
1. Clearly states the core hypothesis
2. Describes the analysis procedure
3. Presents the key findings with detailed figure captions
4. Concludes with the main biological implications

The report should be written in a clear, scientific style suitable for a research paper. For each figure you select, provide:
- A detailed caption explaining the biological significance
- The key patterns or relationships shown
- The biological implications of the findings

Format your response as follows:
[INTRODUCTION]
[Your introduction text]

[SELECTED_FIGURES]
Figure 1: [Figure caption and biological significance]
Figure 2: [Figure caption and biological significance]
...

[CONCLUSION]
[Your conclusion text]"""
        
        return prompt
    
    def _parse_report_content(self, report_content):
        """Parse the LLM response into sections"""
        sections = {
            'introduction': '',
            'figures': [],
            'conclusion': ''
        }
        
        current_section = 'introduction'
        for line in report_content.split('\n'):
            if '[SELECTED_FIGURES]' in line:
                current_section = 'figures'
                continue
            elif '[CONCLUSION]' in line:
                current_section = 'conclusion'
                continue
            
            if current_section == 'figures' and line.strip():
                # Handle different figure caption formats
                if line.strip().startswith('Figure'):
                    sections['figures'].append(line.strip())
                elif line.strip().startswith('Fig.'):
                    sections['figures'].append(line.strip())
            else:
                sections[current_section] += line + '\n'
        
        return sections
    
    def _create_pdf_report(self, pdf_path, report_sections, figures):
        """Create a PDF report with the selected figures and their captions"""
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        temp_files = []  # Keep track of temporary files
        
        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30
        )
        story.append(Paragraph("Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Add introduction
        story.append(Paragraph("Introduction", styles['Heading2']))
        story.append(Paragraph(report_sections['introduction'], styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Add figures and captions
        story.append(Paragraph("Key Findings", styles['Heading2']))
        for figure_caption in report_sections['figures']:
            try:
                # Extract figure number from caption, handling different formats
                fig_num = 0
                if ':' in figure_caption:
                    # Handle "Figure X:" format
                    fig_num = int(figure_caption.split(':')[0].split()[-1]) - 1
                elif '.' in figure_caption:
                    # Handle "Fig. X." format
                    fig_num = int(figure_caption.split('.')[1].strip()) - 1
                
                if 0 <= fig_num < len(figures):
                    # Save figure to temporary file
                    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    temp_files.append(temp_file.name)  # Keep track of the file
                    
                    img_data = base64.b64decode(figures[fig_num]['data'])
                    temp_file.write(img_data)
                    temp_file.close()  # Close the file but don't delete it
                    
                    # Add figure to PDF
                    img = RLImage(temp_file.name, width=6*inch, height=4*inch)
                    story.append(img)
                    
                    # Add caption
                    story.append(Paragraph(figure_caption, styles['Normal']))
                    story.append(Spacer(1, 12))
                else:
                    print(f"Warning: Figure number {fig_num + 1} out of range")
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse figure number from caption: {figure_caption}")
                continue
        
        # Add conclusion
        story.append(Paragraph("Conclusion", styles['Heading2']))
        story.append(Paragraph(report_sections['conclusion'], styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        # Clean up temporary files after PDF is built
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file}: {str(e)}")
    
    def _generate_report_with_llm(self, prompt, figures):
        """Generate the report using the LLM with vision capabilities"""
        messages = [
            {"role": "system", "content": "You are a scientific report writer specializing in single-cell analysis. Your task is to analyze the provided content and generate a comprehensive report that clearly communicates the hypothesis, methods, and findings. Focus on selecting the most biologically significant figures and providing detailed captions that explain their importance."},
            {"role": "user", "content": prompt}
        ]
        
        # Add figures to the messages
        for figure in figures:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Figure Context:\nCode:\n{figure['context']['code']}\n\nOutput:\n{figure['context']['text_output']}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{figure['data']}"
                        }
                    }
                ]
            })
        
        # Generate the report
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=2000
        )
        
        return response.choices[0].message.content

def main():
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate a report from a Jupyter notebook')
    parser.add_argument('notebook_path', help='Path to the Jupyter notebook')
    parser.add_argument('--output-dir', default='reports', help='Directory to save the report')
    parser.add_argument('--model', default='gpt-4o', help='OpenAI model to use')
    
    args = parser.parse_args()
    
    reporter = NotebookReporter(os.getenv("OPENAI_API_KEY"), args.model)
    report_path = reporter.generate_report(args.notebook_path, args.output_dir)
    print(f"Report generated successfully: {report_path}")

if __name__ == "__main__":
    main() 