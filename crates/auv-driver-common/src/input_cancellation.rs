//! Cancellation shared by synchronous input operations.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// A transport-owned signal that lets blocking input work release held state.
#[derive(Default)]
pub struct InputCancellation {
  cancelled: AtomicBool,
  waiters: Mutex<Vec<Box<dyn Fn() + Send + Sync>>>,
}

impl InputCancellation {
  pub fn cancel(&self) {
    self.cancelled.store(true, Ordering::Release);
    for wake in self.waiters.lock().unwrap().iter() {
      wake();
    }
  }

  pub fn is_cancelled(&self) -> bool {
    self.cancelled.load(Ordering::Acquire)
  }

  /// Register a waiter's wakeup while its operation is active.
  pub(crate) fn register_wakeup(&self, wake: impl Fn() + Send + Sync + 'static) {
    self.waiters.lock().unwrap().push(Box::new(wake));
  }
}

thread_local! {
  static INPUT_CANCELLATION: RefCell<Option<Arc<InputCancellation>>> = const { RefCell::new(None) };
}

pub fn current_input_cancellation() -> Option<Arc<InputCancellation>> {
  INPUT_CANCELLATION.with(|value| value.borrow().clone())
}

/// Bind the transport's cancellation signal to synchronous native input work.
pub fn with_input_cancellation<T>(cancelled: Arc<InputCancellation>, action: impl FnOnce() -> T) -> T {
  struct Restore(Option<Arc<InputCancellation>>);

  impl Drop for Restore {
    fn drop(&mut self) {
      INPUT_CANCELLATION.with(|value| *value.borrow_mut() = self.0.take());
    }
  }

  let _restore = Restore(INPUT_CANCELLATION.with(|value| value.replace(Some(cancelled))));
  action()
}
