"""
Main application window for the image annotation tool.

This provides a GUI for testing VLMChat pipelines with scenario files.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Dict, Any, Optional
from PIL import Image, ImageTk, ImageDraw
import yaml
import os

from .mock_data import get_mock_detections, get_mock_prompts, get_mock_scenario, MockDetection

# Try to import pipeline integration (graceful fallback to mock mode)
try:
    from .pipeline_integration import PipelineAdapter
    from .scenario_parser import ScenarioParser
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    PipelineAdapter = None
    ScenarioParser = None


class AnnotationTool:
    """Main application window for image annotation."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("VLMChat Annotation Tool")
        self.root.geometry("1400x900")
        
        # Application state
        self.current_scenario: Optional[Dict[str, Any]] = None
        self.current_scenario_id: Optional[str] = None
        self.scenario_parser: Optional[Any] = None  # ScenarioParser instance
        self.current_detections: List[MockDetection] = []
        self.current_prompts: Dict[str, str] = {}
        self.current_image: Optional[Image.Image] = None
        self.detection_depth = 2  # Default depth for detection tree
        
        # Pipeline integration (if available)
        self.pipeline_adapter = None
        if PIPELINE_AVAILABLE:
            try:
                self.pipeline_adapter = PipelineAdapter()
                self.status_mode = "Pipeline Mode"
            except Exception as e:
                self.pipeline_adapter = None
                self.status_mode = f"Mock Mode (Pipeline error: {e})"
        else:
            self.status_mode = "Mock Mode (Pipeline not available)"
        
        # UI Components
        self.setup_ui()
        
        # Load mock data for initial testing
        self.load_mock_data()
    
    def setup_ui(self) -> None:
        """Set up the main UI layout."""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Scenario", command=self.load_scenario)
        file_menu.add_command(label="Load Image", command=self.load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Save Scenario", command=self.save_scenario)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Main container with three panels
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel: Image display
        self.setup_image_panel(main_container)
        
        # Middle panel: Detection tree
        self.setup_detection_panel(main_container)
        
        # Right panel: Prompt editor
        self.setup_prompt_panel(main_container)
        
        # Bottom panel: Controls
        self.setup_control_panel()
    
    def setup_image_panel(self, parent: ttk.PanedWindow) -> None:
        """Set up the image display panel."""
        image_frame = ttk.LabelFrame(parent, text="Image Display", padding=10)
        parent.add(image_frame, weight=3)
        
        # Canvas for image display
        self.image_canvas = tk.Canvas(image_frame, bg="gray", width=600, height=600)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        h_scroll = ttk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        v_scroll = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        self.image_canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        
        # Status label
        self.image_status = ttk.Label(image_frame, text="No image loaded")
        self.image_status.pack(side=tk.BOTTOM, pady=5)
    
    def setup_detection_panel(self, parent: ttk.PanedWindow) -> None:
        """Set up the detection tree panel."""
        detection_frame = ttk.LabelFrame(parent, text="Detections", padding=10)
        parent.add(detection_frame, weight=2)
        
        # Depth selector
        depth_frame = ttk.Frame(detection_frame)
        depth_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(depth_frame, text="Display Depth:").pack(side=tk.LEFT, padx=5)
        self.depth_var = tk.IntVar(value=self.detection_depth)
        depth_spin = ttk.Spinbox(depth_frame, from_=1, to=10, 
                                  textvariable=self.depth_var, width=5,
                                  command=self.update_detection_tree)
        depth_spin.pack(side=tk.LEFT, padx=5)
        
        # Detection tree
        tree_frame = ttk.Frame(detection_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.detection_tree = ttk.Treeview(tree_frame, columns=("id", "category", "confidence", "box"),
                                           show="tree headings", selectmode="browse")
        
        self.detection_tree.heading("#0", text="Detection")
        self.detection_tree.heading("id", text="ID")
        self.detection_tree.heading("category", text="Category")
        self.detection_tree.heading("confidence", text="Conf")
        self.detection_tree.heading("box", text="Bounding Box")
        
        self.detection_tree.column("#0", width=150)
        self.detection_tree.column("id", width=50)
        self.detection_tree.column("category", width=100)
        self.detection_tree.column("confidence", width=60)
        self.detection_tree.column("box", width=150)
        
        # Scrollbar for tree
        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.detection_tree.yview)
        self.detection_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.detection_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event to highlight detection on image
        self.detection_tree.bind("<<TreeviewSelect>>", self.on_detection_select)
    
    def setup_prompt_panel(self, parent: ttk.PanedWindow) -> None:
        """Set up the prompt editor panel."""
        prompt_frame = ttk.LabelFrame(parent, text="Prompts", padding=10)
        parent.add(prompt_frame, weight=2)
        
        # Prompt list
        list_frame = ttk.Frame(prompt_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.prompt_listbox = tk.Listbox(list_frame, height=10)
        self.prompt_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.prompt_listbox.bind("<<ListboxSelect>>", self.on_prompt_select)
        
        list_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.prompt_listbox.yview)
        self.prompt_listbox.configure(yscrollcommand=list_scroll.set)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Prompt editor
        editor_label = ttk.Label(prompt_frame, text="Edit Prompt:")
        editor_label.pack(fill=tk.X)
        
        self.prompt_text = tk.Text(prompt_frame, height=8, wrap=tk.WORD)
        self.prompt_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Prompt editor buttons
        button_frame = ttk.Frame(prompt_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Update Prompt", 
                  command=self.update_prompt).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Add Prompt", 
                  command=self.add_prompt).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Delete Prompt", 
                  command=self.delete_prompt).pack(side=tk.LEFT, padx=5)
    
    def setup_control_panel(self) -> None:
        """Set up the bottom control panel."""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        # Control buttons
        ttk.Button(control_frame, text="Run Pipeline", 
                  command=self.run_pipeline).pack(side=tk.RIGHT, padx=5)
        ttk.Button(control_frame, text="Update Scenario", 
                  command=self.update_scenario).pack(side=tk.RIGHT, padx=5)
        ttk.Button(control_frame, text="Load Mock Data", 
                  command=self.load_mock_data).pack(side=tk.RIGHT, padx=5)
    
    def load_mock_data(self) -> None:
        """Load mock data for UI testing."""
        self.current_detections = get_mock_detections()
        self.current_prompts = get_mock_prompts()
        self.current_scenario = get_mock_scenario()
        
        # Create a mock image
        self.current_image = Image.new("RGB", (1920, 1080), color="lightgray")
        draw = ImageDraw.Draw(self.current_image)
        
        # Draw mock detections
        for det in self.current_detections:
            self._draw_detection_recursive(draw, det)
        
        # Update UI
        self.update_detection_tree()
        self.update_prompt_list()
        self.display_image()
        self.status_label.config(text=f"Mock data loaded | {self.status_mode}")
    
    def _draw_detection_recursive(self, draw: ImageDraw.Draw, detection: MockDetection, 
                                   color: str = "red", depth: int = 0) -> None:
        """Recursively draw detections and their children."""
        x1, y1, x2, y2 = detection.box
        
        # Vary color by depth
        colors = ["red", "blue", "green", "orange", "purple"]
        color = colors[depth % len(colors)]
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        label = f"{detection.object_category} ({detection.conf:.2f})"
        draw.text((x1, y1 - 15), label, fill=color)
        
        # Draw children
        for child in detection.children:
            self._draw_detection_recursive(draw, child, depth=depth + 1)
    
    def update_detection_tree(self) -> None:
        """Update the detection tree view."""
        # Clear existing tree
        for item in self.detection_tree.get_children():
            self.detection_tree.delete(item)
        
        # Get current depth
        self.detection_depth = self.depth_var.get()
        
        # Populate tree
        for detection in self.current_detections:
            self._add_detection_to_tree("", detection, 0, self.detection_depth)
    
    def _add_detection_to_tree(self, parent: str, detection: MockDetection, 
                                current_depth: int, max_depth: int) -> None:
        """Recursively add detection to tree."""
        if current_depth >= max_depth:
            return
        
        # Add detection node
        node_id = self.detection_tree.insert(
            parent, 
            tk.END, 
            text=f"Detection {detection.id}",
            values=(
                detection.id,
                detection.object_category,
                f"{detection.conf:.2f}",
                f"({detection.box[0]}, {detection.box[1]}, {detection.box[2]}, {detection.box[3]})"
            ),
            tags=(str(detection.id),)
        )
        
        # Add children
        for child in detection.children:
            self._add_detection_to_tree(node_id, child, current_depth + 1, max_depth)
    
    def update_prompt_list(self) -> None:
        """Update the prompt list view."""
        self.prompt_listbox.delete(0, tk.END)
        for name in self.current_prompts.keys():
            self.prompt_listbox.insert(tk.END, name)
    
    def display_image(self) -> None:
        """Display the current image on the canvas."""
        if self.current_image is None:
            return
        
        # Resize image to fit canvas while maintaining aspect ratio
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        # Use a reasonable default if canvas hasn't been drawn yet
        if canvas_width <= 1:
            canvas_width = 600
        if canvas_height <= 1:
            canvas_height = 600
        
        img_width, img_height = self.current_image.size
        
        # Calculate scaling factor
        scale_w = canvas_width / img_width
        scale_h = canvas_height / img_height
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        resized_img = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo_image = ImageTk.PhotoImage(resized_img)
        
        # Clear canvas and display image
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        self.image_canvas.config(scrollregion=self.image_canvas.bbox(tk.ALL))
        
        self.image_status.config(text=f"Image: {img_width}x{img_height}")
    
    def on_detection_select(self, event: tk.Event) -> None:
        """Handle detection selection in tree."""
        selection = self.detection_tree.selection()
        if selection:
            item = self.detection_tree.item(selection[0])
            det_id = item['values'][0]
            self.status_label.config(text=f"Selected Detection ID: {det_id}")
    
    def on_prompt_select(self, event: tk.Event) -> None:
        """Handle prompt selection in list."""
        selection = self.prompt_listbox.curselection()
        if selection:
            prompt_name = self.prompt_listbox.get(selection[0])
            prompt_text = self.current_prompts.get(prompt_name, "")
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert("1.0", prompt_text)
            self.status_label.config(text=f"Editing: {prompt_name}")
    
    def update_prompt(self) -> None:
        """Update the selected prompt."""
        selection = self.prompt_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a prompt to update")
            return
        
        prompt_name = self.prompt_listbox.get(selection[0])
        new_text = self.prompt_text.get("1.0", tk.END).strip()
        
        self.current_prompts[prompt_name] = new_text
        self.status_label.config(text=f"Updated: {prompt_name}")
        messagebox.showinfo("Success", f"Prompt '{prompt_name}' updated")
    
    def add_prompt(self) -> None:
        """Add a new prompt."""
        # Simple dialog for new prompt name
        name = tk.simpledialog.askstring("Add Prompt", "Enter prompt name:")
        if name:
            if name in self.current_prompts:
                messagebox.showwarning("Duplicate", "Prompt name already exists")
                return
            
            self.current_prompts[name] = ""
            self.update_prompt_list()
            self.status_label.config(text=f"Added: {name}")
    
    def delete_prompt(self) -> None:
        """Delete the selected prompt."""
        selection = self.prompt_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a prompt to delete")
            return
        
        prompt_name = self.prompt_listbox.get(selection[0])
        
        if messagebox.askyesno("Confirm Delete", f"Delete prompt '{prompt_name}'?"):
            del self.current_prompts[prompt_name]
            self.update_prompt_list()
            self.prompt_text.delete("1.0", tk.END)
            self.status_label.config(text=f"Deleted: {prompt_name}")
    
    def load_scenario(self) -> None:
        """Load a scenario file."""
        filename = filedialog.askopenfilename(
            title="Load Scenario",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'r') as f:
                scenario_data = yaml.safe_load(f)
            
            # Parse scenario if parser available
            if ScenarioParser:
                self.scenario_parser = ScenarioParser(scenario_data)
                scenario_ids = self.scenario_parser.get_all_scenario_ids()
                
                # If multiple scenarios, ask which one to load
                if len(scenario_ids) > 1:
                    # For now, load the first one
                    # TODO: Add scenario selector dialog
                    scenario_id = scenario_ids[0]
                    self.current_scenario_id = scenario_id
                    self.current_scenario = self.scenario_parser.get_scenario_by_id(scenario_id)
                    
                    # Extract prompts from scenario
                    self.current_prompts = self.scenario_parser.extract_prompts(self.current_scenario)
                    self.update_prompt_list()
                    
                    messagebox.showinfo("Scenario Loaded", 
                                      f"Loaded scenario: {scenario_id}\n"
                                      f"({len(scenario_ids)} scenarios in file)\n"
                                      f"Prompts: {len(self.current_prompts)}")
                else:
                    # Single scenario
                    self.current_scenario_id = scenario_ids[0] if scenario_ids else None
                    self.current_scenario = self.scenario_parser.get_scenario_by_id(self.current_scenario_id) if self.current_scenario_id else None
                    
                    if self.current_scenario:
                        self.current_prompts = self.scenario_parser.extract_prompts(self.current_scenario)
                        self.update_prompt_list()
                        
                        messagebox.showinfo("Scenario Loaded", 
                                          f"Loaded scenario: {self.current_scenario_id}\n"
                                          f"Prompts: {len(self.current_prompts)}")
            else:
                # Fallback: just store the raw data
                self.current_scenario = scenario_data
                messagebox.showinfo("Scenario Loaded", 
                                  "Scenario loaded (parser not available)")
            
            self.status_label.config(text=f"Loaded: {os.path.basename(filename)} | {self.status_mode}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load scenario: {e}")
            import traceback
            traceback.print_exc()
    
    def load_image(self) -> None:
        """Load an image file."""
        filename = filedialog.askopenfilename(
            title="Load Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            self.current_image = Image.open(filename)
            self.display_image()
            self.status_label.config(text=f"Loaded image: {os.path.basename(filename)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def save_scenario(self) -> None:
        """Save the current scenario to file."""
        if self.current_scenario is None:
            messagebox.showwarning("No Scenario", "No scenario to save")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Scenario",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # Update scenario with current prompts if parser available
            if self.scenario_parser and self.current_scenario_id:
                self.scenario_parser.update_scenario_prompts(self.current_scenario_id, 
                                                            self.current_prompts)
                scenario_data = self.scenario_parser.to_dict()
            else:
                scenario_data = self.current_scenario
            
            with open(filename, 'w') as f:
                yaml.dump(scenario_data, f, default_flow_style=False, sort_keys=False)
            
            self.status_label.config(text=f"Saved: {os.path.basename(filename)} | {self.status_mode}")
            messagebox.showinfo("Success", "Scenario saved with updated prompts")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save scenario: {e}")
            import traceback
            traceback.print_exc()
    
    def run_pipeline(self) -> None:
        """Run the pipeline with current configuration."""
        # TODO: Integrate with actual VLMChat pipeline
        self.status_label.config(text="Running pipeline... (mock)")
        self.root.update()
        
        # Simulate pipeline execution
        import time
        time.sleep(0.5)
        
        # For now, just reload mock data
        self.load_mock_data()
        self.status_label.config(text="Pipeline complete (mock)")
        messagebox.showinfo("Pipeline", "Pipeline execution complete (mock mode)")
    
    def update_scenario(self) -> None:
        """Update the scenario with current prompts and results."""
        if self.current_scenario is None:
            messagebox.showwarning("No Scenario", "No scenario loaded")
            return
        
        # TODO: Update scenario with current prompts and detection results
        self.status_label.config(text="Scenario updated (mock)")
        messagebox.showinfo("Success", "Scenario updated with current prompts (mock mode)")


def main():
    """Main entry point for the annotation tool."""
    root = tk.Tk()
    
    # Import simpledialog for add_prompt functionality
    import tkinter.simpledialog
    tk.simpledialog = tkinter.simpledialog
    
    app = AnnotationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
