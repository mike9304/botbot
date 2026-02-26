//! Fixed-capacity ring buffer with time-based expiry.
//!
//! Stores [`TimestampedEvent`] items in a circular buffer. Events older than the
//! configured TTL are considered expired and excluded from iteration, though they
//! are only physically overwritten when the buffer wraps around.

use chrono::{DateTime, Duration, Utc};

// ---------------------------------------------------------------------------
// TimestampedEvent
// ---------------------------------------------------------------------------

/// An event stored in the ring buffer, paired with its arrival timestamp.
#[derive(Debug, Clone)]
pub struct TimestampedEvent<T> {
    pub event: T,
    pub timestamp: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// RingBuffer
// ---------------------------------------------------------------------------

/// A fixed-capacity circular buffer that tracks events with timestamps.
///
/// # Time-based expiry
///
/// Events older than `ttl` are considered expired. They are still physically
/// present in the buffer but are skipped during iteration ([`iter_active`]).
/// When the buffer is full the oldest slot is silently overwritten.
#[derive(Debug)]
pub struct RingBuffer<T> {
    /// Internal storage.
    buf: Vec<Option<TimestampedEvent<T>>>,
    /// Maximum number of elements.
    capacity: usize,
    /// Index where the next element will be written.
    head: usize,
    /// Total number of elements currently stored (may include expired ones).
    len: usize,
    /// Time-to-live for events (default: 5 minutes).
    ttl: Duration,
}

impl<T: Clone> RingBuffer<T> {
    /// Create a new ring buffer with the given capacity and a 5-minute TTL.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "RingBuffer capacity must be > 0");
        Self {
            buf: vec![None; capacity],
            capacity,
            head: 0,
            len: 0,
            ttl: Duration::minutes(5),
        }
    }

    /// Create a ring buffer with a custom TTL.
    pub fn with_ttl(capacity: usize, ttl: Duration) -> Self {
        assert!(capacity > 0, "RingBuffer capacity must be > 0");
        Self {
            buf: vec![None; capacity],
            capacity,
            head: 0,
            len: 0,
            ttl,
        }
    }

    /// Push an event with the current UTC timestamp.
    pub fn push(&mut self, event: T) {
        self.push_at(event, Utc::now());
    }

    /// Push an event with an explicit timestamp (useful for testing).
    pub fn push_at(&mut self, event: T, timestamp: DateTime<Utc>) {
        self.buf[self.head] = Some(TimestampedEvent { event, timestamp });
        self.head = (self.head + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Iterate over all non-expired events in chronological order (oldest first).
    pub fn iter_active(&self) -> impl Iterator<Item = &TimestampedEvent<T>> {
        let now = Utc::now();
        self.iter_active_at(now)
    }

    /// Iterate over events that are active relative to a given reference time.
    pub fn iter_active_at(
        &self,
        reference: DateTime<Utc>,
    ) -> impl Iterator<Item = &TimestampedEvent<T>> {
        let cutoff = reference - self.ttl;
        self.iter_all().filter(move |e| e.timestamp >= cutoff)
    }

    /// Iterate over **all** stored events in chronological order, including expired ones.
    pub fn iter_all(&self) -> impl Iterator<Item = &TimestampedEvent<T>> {
        let start = if self.len < self.capacity {
            0
        } else {
            self.head // oldest element when buffer is full
        };

        let capacity = self.capacity;
        let len = self.len;
        let buf = &self.buf;

        (0..len).filter_map(move |i| {
            let idx = (start + i) % capacity;
            buf[idx].as_ref()
        })
    }

    /// Number of events currently stored (including expired).
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum capacity of the buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of non-expired events.
    pub fn active_count(&self) -> usize {
        self.iter_active().count()
    }

    /// Clear all events.
    pub fn clear(&mut self) {
        for slot in self.buf.iter_mut() {
            *slot = None;
        }
        self.head = 0;
        self.len = 0;
    }

    /// Collect all active events into a Vec, consuming nothing.
    pub fn active_events(&self) -> Vec<&TimestampedEvent<T>> {
        self.iter_active().collect()
    }

    /// Get the most recently pushed event, if any.
    pub fn latest(&self) -> Option<&TimestampedEvent<T>> {
        if self.len == 0 {
            return None;
        }
        let idx = if self.head == 0 {
            self.capacity - 1
        } else {
            self.head - 1
        };
        self.buf[idx].as_ref()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_push_and_len() {
        let mut rb: RingBuffer<i32> = RingBuffer::new(4);
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);

        rb.push(1);
        rb.push(2);
        rb.push(3);
        assert_eq!(rb.len(), 3);
        assert!(!rb.is_empty());
    }

    #[test]
    fn test_capacity_wrap() {
        let mut rb: RingBuffer<i32> = RingBuffer::new(3);
        rb.push(1);
        rb.push(2);
        rb.push(3);
        rb.push(4); // overwrites slot 0

        assert_eq!(rb.len(), 3);
        let all: Vec<i32> = rb.iter_all().map(|e| e.event.clone()).collect();
        assert_eq!(all, vec![2, 3, 4]);
    }

    #[test]
    fn test_chronological_order() {
        let mut rb: RingBuffer<&str> = RingBuffer::new(5);
        let base = Utc::now();
        rb.push_at("a", base);
        rb.push_at("b", base + Duration::seconds(1));
        rb.push_at("c", base + Duration::seconds(2));

        let events: Vec<&str> = rb.iter_all().map(|e| e.event).collect();
        assert_eq!(events, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_time_expiry() {
        let mut rb: RingBuffer<&str> = RingBuffer::with_ttl(10, Duration::minutes(5));
        let now = Utc::now();

        // Push an old event (6 minutes ago)
        rb.push_at("old", now - Duration::minutes(6));
        // Push a recent event
        rb.push_at("new", now);

        assert_eq!(rb.len(), 2);
        let active: Vec<&str> = rb.iter_active_at(now).map(|e| e.event).collect();
        assert_eq!(active, vec!["new"]);
    }

    #[test]
    fn test_active_count() {
        let mut rb: RingBuffer<i32> = RingBuffer::with_ttl(10, Duration::minutes(5));
        let now = Utc::now();

        rb.push_at(1, now - Duration::minutes(10));
        rb.push_at(2, now - Duration::minutes(3));
        rb.push_at(3, now);

        // When checking at `now`, only events within 5 minutes are active.
        let count = rb.iter_active_at(now).count();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_latest() {
        let mut rb: RingBuffer<i32> = RingBuffer::new(4);
        assert!(rb.latest().is_none());

        rb.push(10);
        assert_eq!(rb.latest().unwrap().event, 10);

        rb.push(20);
        assert_eq!(rb.latest().unwrap().event, 20);
    }

    #[test]
    fn test_clear() {
        let mut rb: RingBuffer<i32> = RingBuffer::new(4);
        rb.push(1);
        rb.push(2);
        rb.clear();
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
    }

    #[test]
    fn test_wrap_around_multiple_times() {
        let mut rb: RingBuffer<i32> = RingBuffer::new(2);
        for i in 0..10 {
            rb.push(i);
        }
        assert_eq!(rb.len(), 2);
        let all: Vec<i32> = rb.iter_all().map(|e| e.event).collect();
        assert_eq!(all, vec![8, 9]);
    }

    #[test]
    fn test_active_events_convenience() {
        let mut rb: RingBuffer<i32> = RingBuffer::new(5);
        rb.push(1);
        rb.push(2);
        let events = rb.active_events();
        assert_eq!(events.len(), 2);
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn test_zero_capacity_panics() {
        let _: RingBuffer<i32> = RingBuffer::new(0);
    }

    #[test]
    fn test_latest_after_wrap() {
        let mut rb: RingBuffer<i32> = RingBuffer::new(3);
        rb.push(1);
        rb.push(2);
        rb.push(3);
        rb.push(4); // wraps, head now at 1
        assert_eq!(rb.latest().unwrap().event, 4);
    }
}
