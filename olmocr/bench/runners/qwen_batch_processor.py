"""
GPU memory optimized batch processing for Qwen2.5-VL silver data generation.
"""

import gc
import torch
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


class QwenBatchProcessor:
    """Optimized batch processor for Qwen2.5-VL with memory management."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: Optional[str] = None,
        max_memory_usage: float = 0.85,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_memory_usage = max_memory_usage
        
        self.model = None
        self.processor = None
        
    def _load_model(self):
        """Load model with memory optimization."""
        if self.model is None:
            print(f"Loading {self.model_name}...")
            
            # Clear any existing GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2"
            ).eval()
            
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            print("Model loaded successfully")
    
    def _unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
    @contextmanager
    def model_context(self):
        """Context manager for model loading/unloading."""
        try:
            self._load_model()
            yield
        finally:
            self._unload_model()
    
    def check_memory_usage(self) -> bool:
        """Check if GPU memory usage is under threshold."""
        if not torch.cuda.is_available():
            return True
            
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        usage = allocated / total
        
        return usage < self.max_memory_usage
    
    def process_batch(
        self, 
        pdf_paths_and_pages: List[tuple],
        prompt_template: str = "yaml",
        response_template: str = "yaml",
        temperature: float = 0.1,
        max_new_tokens: int = 3000,
        target_image_dim: int = 2048
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Process a batch of PDF pages.
        
        Args:
            pdf_paths_and_pages: List of (pdf_path, page_num) tuples
            prompt_template: Prompt template to use
            response_template: Response template to use
            temperature: Generation temperature
            max_new_tokens: Maximum new tokens
            target_image_dim: Target image dimension
            
        Returns:
            List of results (same order as input)
        """
        results = []
        
        with self.model_context():
            for pdf_path, page_num in pdf_paths_and_pages:
                try:
                    # Check memory before processing each item
                    if not self.check_memory_usage():
                        print("Warning: High GPU memory usage detected")
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    result = self._process_single_page(
                        pdf_path=pdf_path,
                        page_num=page_num,
                        prompt_template=prompt_template,
                        response_template=response_template,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        target_image_dim=target_image_dim
                    )
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error processing {pdf_path} page {page_num}: {e}")
                    results.append(None)
                    
                # Clean up after each page
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        return results
    
    def _process_single_page(
        self,
        pdf_path: str,
        page_num: int,
        prompt_template: str,
        response_template: str,
        temperature: float,
        max_new_tokens: int,
        target_image_dim: int
    ) -> Optional[str]:
        """Process a single PDF page (internal method)."""
        from olmocr.bench.runners.run_transformers import run_transformers
        
        # Use the existing run_transformers function but with our loaded model
        # This is a bit of a hack but avoids duplicating the logic
        global _cached_model, _cached_processor
        from olmocr.bench.runners import run_transformers as rt_module
        
        # Temporarily override the global cache
        old_model = getattr(rt_module, '_cached_model', None)
        old_processor = getattr(rt_module, '_cached_processor', None)
        
        rt_module._cached_model = self.model
        rt_module._cached_processor = self.processor
        
        try:
            result = run_transformers(
                pdf_path=pdf_path,
                page_num=page_num,
                model_name=self.model_name,  # Will be ignored due to cache
                temperature=temperature,
                target_longest_image_dim=target_image_dim,
                prompt_template=prompt_template,
                response_template=response_template
            )
            return result
        finally:
            # Restore the old cache
            rt_module._cached_model = old_model
            rt_module._cached_processor = old_processor
