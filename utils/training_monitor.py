#!/usr/bin/env python
# coding=utf-8

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Training monitor for tracking metrics and generating plots"""
    
    def __init__(self, output_dir, enable_monitoring=True):
        """
        Args:
            output_dir: Directory to save monitoring outputs
            enable_monitoring: Whether to enable monitoring (can be disabled via config)
        """
        self.output_dir = output_dir
        self.enable_monitoring = enable_monitoring
        
        if self.enable_monitoring:
            os.makedirs(output_dir, exist_ok=True)
            
        # Training metrics tracking
        self.metrics = {
            'steps': [],
            'esd_loss': [],
            'attn_loss': [],
            'lora_loss': [],
            'infonce_loss': [],
            'preserve_loss': [],
            'total_loss': [],
            'learning_rate': []
        }
    
    def log_step(self, step, esd_loss, attn_loss, preserve_loss, total_loss, lr):
        """Log metrics for a single training step"""
        if not self.enable_monitoring:
            return
            
        self.metrics['steps'].append(step)
        self.metrics['esd_loss'].append(esd_loss)
        self.metrics['attn_loss'].append(attn_loss)
        self.metrics['preserve_loss'].append(preserve_loss)
        self.metrics['total_loss'].append(total_loss)
        self.metrics['learning_rate'].append(lr)
    
    def save_metrics(self):
        """Save training metrics to JSON file"""
        if not self.enable_monitoring:
            return
            
        metrics_path = os.path.join(self.output_dir, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Training metrics saved to {metrics_path}")
    
    def generate_plots(self):
        """Generate and save training loss plots"""
        if not self.enable_monitoring:
            return
            
        if not self.metrics['steps']:
            logger.warning("No metrics to plot")
            return
            
        # Generate main training curves
        self._generate_main_plots()
        
        # Generate combined loss plot
        self._generate_combined_plot()
        
        # Print training summary
        self._print_training_summary()
    
    def _generate_main_plots(self):
        """Generate 4-subplot training curves (2x2 layout)"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        steps = self.metrics['steps']
        
        # 1. ESD Loss (top-left)
        axes[0, 0].plot(steps, self.metrics['esd_loss'], 'r-', linewidth=2, label='ESD Loss')
        axes[0, 0].set_title('ESD Loss')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. Attention Loss (top-right)
        axes[0, 1].plot(steps, self.metrics['attn_loss'], 'g-', linewidth=2, label='Attention Loss')
        axes[0, 1].set_title('Attention Loss')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 3. Preserve Loss (bottom-left)
        axes[1, 0].plot(steps, self.metrics['preserve_loss'], 'b-', linewidth=2, label='Preserve Loss')
        axes[1, 0].set_title('Preserve Loss')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # 4. Total Loss (bottom-right)
        axes[1, 1].plot(steps, self.metrics['total_loss'], 'k-', linewidth=2, label='Total Loss')
        axes[1, 1].set_title('Total Loss')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {plot_path}")
        plt.close()
    
    def _generate_combined_plot(self):
        """Generate combined loss plot - All losses"""
        steps = self.metrics['steps']
        
        plt.figure(figsize=(12, 6))
        
        # Plot all losses
        plt.plot(steps, self.metrics['esd_loss'], 'r-', linewidth=2, label='ESD Loss')
        plt.plot(steps, self.metrics['attn_loss'], 'g-', linewidth=2, label='Attention Loss')
        plt.plot(steps, self.metrics['preserve_loss'], 'b-', linewidth=2, label='Preserve Loss')
        plt.plot(steps, self.metrics['total_loss'], 'k--', linewidth=2.5, label='Total Loss')
        
        plt.title('Training Losses Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        combined_path = os.path.join(self.output_dir, "loss_curves_combined.png")
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        logger.info(f"Combined loss curves saved to {combined_path}")
        plt.close()
    
    def _print_training_summary(self):
        """Print training summary statistics - All losses"""
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        
        total_steps = len(self.metrics['steps'])
        
        # Calculate combined loss (forget loss)
        forget_loss = np.array(self.metrics['esd_loss']) + np.array(self.metrics['attn_loss'])
        
        # Final values
        final_forget_loss = forget_loss[-1]
        final_esd_loss = self.metrics['esd_loss'][-1]
        final_attn_loss = self.metrics['attn_loss'][-1]
        final_preserve_loss = self.metrics['preserve_loss'][-1]
        final_total_loss = self.metrics['total_loss'][-1]
        
        # Average values
        avg_forget_loss = np.mean(forget_loss)
        avg_esd_loss = np.mean(self.metrics['esd_loss'])
        avg_attn_loss = np.mean(self.metrics['attn_loss'])
        avg_preserve_loss = np.mean(self.metrics['preserve_loss'])
        avg_total_loss = np.mean(self.metrics['total_loss'])
        
        # Calculate trends (last 20% vs first 20%)
        n_20_percent = max(1, total_steps // 5)
        first_20_esd = np.mean(self.metrics['esd_loss'][:n_20_percent])
        last_20_esd = np.mean(self.metrics['esd_loss'][-n_20_percent:])
        first_20_attn = np.mean(self.metrics['attn_loss'][:n_20_percent])
        last_20_attn = np.mean(self.metrics['attn_loss'][-n_20_percent:])
        first_20_preserve = np.mean(self.metrics['preserve_loss'][:n_20_percent])
        last_20_preserve = np.mean(self.metrics['preserve_loss'][-n_20_percent:])
        
        esd_improvement = ((first_20_esd - last_20_esd) / first_20_esd) * 100 if first_20_esd > 0 else 0
        attn_improvement = ((first_20_attn - last_20_attn) / first_20_attn) * 100 if first_20_attn > 0 else 0
        preserve_improvement = ((first_20_preserve - last_20_preserve) / first_20_preserve) * 100 if first_20_preserve > 0 else 0
        
        logger.info(f"Total Steps: {total_steps}")
        logger.info("")
        logger.info("=== ESD LOSS ===")
        logger.info(f"Final ESD Loss: {final_esd_loss:.6f}")
        logger.info(f"Average ESD Loss: {avg_esd_loss:.6f}")
        logger.info(f"ESD Loss Improvement: {esd_improvement:.2f}%")
        logger.info("")
        logger.info("=== ATTENTION LOSS ===")
        logger.info(f"Final Attention Loss: {final_attn_loss:.6f}")
        logger.info(f"Average Attention Loss: {avg_attn_loss:.6f}")
        logger.info(f"Attention Loss Improvement: {attn_improvement:.2f}%")
        logger.info("")
        logger.info("=== FORGET LOSS (ESD + Attention) ===")
        logger.info(f"Final Forget Loss: {final_forget_loss:.6f}")
        logger.info(f"Average Forget Loss: {avg_forget_loss:.6f}")
        logger.info("")
        logger.info("=== PRESERVE LOSS ===")
        logger.info(f"Final Preserve Loss: {final_preserve_loss:.6f}")
        logger.info(f"Average Preserve Loss: {avg_preserve_loss:.6f}")
        logger.info(f"Preserve Loss Improvement: {preserve_improvement:.2f}%")
        logger.info("")
        logger.info("=== TOTAL LOSS ===")
        logger.info(f"Final Total Loss: {final_total_loss:.6f}")
        logger.info(f"Average Total Loss: {avg_total_loss:.6f}")
        logger.info("=" * 60)
    
    def finish_training(self):
        """Call this at the end of training to save metrics and generate plots"""
        if not self.enable_monitoring:
            return
            
        self.save_metrics()
        self.generate_plots()