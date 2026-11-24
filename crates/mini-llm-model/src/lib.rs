//! Model layer for mini-llm.
//!
//! - [`TinyGpt`]: a decoder-only Transformer language model built from autograd
//!   ops, [`TransformerBlock`]s and causal attention.
//! - [`ModelConfig`]: dimensions plus architecture switches (norm, positions,
//!   feed-forward, grouped-query attention, MoE).
//! - [`scaled_dot_product_attention`]: the original transparent single-head
//!   primitive used by the CLI demo and the concept lessons on the web site.

mod attention;
mod block;
mod config;
mod error;
mod gpt;
mod norm;

pub use attention::{
    causal_batch_mask, scaled_dot_product_attention, AttentionCache, AttentionError,
    CausalSelfAttention,
};
pub use block::{BlockCache, Mlp, TransformerBlock};
pub use config::{ConfigError, FeedForwardKind, ModelConfig, NormKind, PositionKind};
pub use error::ModelError;
pub use gpt::TinyGpt;
