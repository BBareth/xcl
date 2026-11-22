//! The console command list's own methods.
//!
//! The console's ID3D12GraphicsCommandList keeps the PC methods (some moved:
//! `real::console_remap`) and holds console extensions from #60 on (as observed on Fortnite):
//! CopyMemoryToMemoryX #60, FillMemoryWith{32,64,128}BitValueX #61–#63, depth bounds #68,
//! Write{32,64}BitValueBottomOfPipeX #69 / #70, Wait{32,64}BitValueX #71 / #72,
//! DispatchX #78, DrawIndexedX #81, CopyResourceX #84, SetPredicationBufferX #85,
//! Write{32,64}BitValueTopOfPipeX #88 / #89, Write32BitValueBottomOfShaderX #90,
//! KickoffX #93, RSSetDepthBiasX #94. A title call of any other extension (GDS, hints…)
//! is logged and does nothing (`real::class_for`).
//!
//! Occlusion queries may nest on the console; the PC allows one active per list. When one
//! begins while another is active, the outer one ends there on the PC and resolves as visible;
//! the inner one is a real query (Fortnite nests its per-object queries in a dummy outer one).
//!
//! Console GPU addresses in unified memory are the title's own addresses; memory operations
//! go to that window's buffer (`real::uma_window_buffer`), which is in COMMON between them.
//! An index buffer at such an address is given to the PC as the window buffer's GPU address.

use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::{transmute_copy, ManuallyDrop};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::device_x::{self, DepthBias};
use super::real;

const PC_GET_TYPE: usize = 8;
const PC_CLOSE: usize = 9;
const PC_RESET: usize = 10;
const PC_DRAW_INDEXED: usize = 13;
const PC_COPY_BUFFER_REGION: usize = 15;
const PC_COPY_TEXTURE_REGION: usize = 16;
const PC_COPY_RESOURCE: usize = 17;
const PC_SET_PIPELINE_STATE: usize = 25;
const PC_RESOURCE_BARRIER: usize = 26;
const PC_IA_SET_INDEX_BUFFER: usize = 43;
const PC_BEGIN_QUERY: usize = 52;
const PC_END_QUERY: usize = 53;
const PC_RESOLVE_QUERY_DATA: usize = 54;
const PC_SET_PREDICATION: usize = 55;
const PC_OM_SET_DEPTH_BOUNDS: usize = 62;
const PC_WRITE_BUFFER_IMMEDIATE: usize = 66;
const PC_QUEUE_EXECUTE_COMMAND_LISTS: usize = 10;

/// Write64BitValueBottomOfPipeX flags 1 / 2, Write64BitValueTopOfPipeX flags 4 / 8: the value
/// written is a GPU / system timestamp
const BOTTOM_TIMESTAMPS: u32 = 0x3;
const TOP_TIMESTAMPS: u32 = 0xC;
/// fills larger than this are not done (one pattern buffer per value, up to this size)
const MAX_FILL: u64 = 256 << 20;
/// Bounded wait timeout for Wait32BitValueX / Wait64BitValueX so an unsatisfied wait never hangs.
const WAIT_TIMEOUT: Duration = Duration::from_secs(2);
/// Cap on tracked per-list written UMA ranges (for same-list write-then-wait detection).
const MAX_WRITTEN_RANGES: usize = 256;

fn once(counter: &AtomicU64, n: u64) -> bool {
    counter.fetch_add(1, Ordering::Relaxed) < n
}

/// a PC method of the list (its runtime's own)
unsafe fn pc<F: Copy>(list: *mut c_void, slot: usize) -> F {
    transmute_copy(&real::orig_fn(list, slot))
}

unsafe fn list_type(list: *mut c_void) -> D3D12_COMMAND_LIST_TYPE {
    pc::<unsafe extern "system" fn(*mut c_void) -> D3D12_COMMAND_LIST_TYPE>(list, PC_GET_TYPE)(list)
}

unsafe fn transition(list: *mut c_void, res: usize, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) {
    let b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: transmute_copy(&res),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    pc::<unsafe extern "system" fn(*mut c_void, u32, *const D3D12_RESOURCE_BARRIER)>(list, PC_RESOURCE_BARRIER)(list, 1, &b);
}

/// `op` with the window buffer `res` as a copy destination (back to COMMON after it)
unsafe fn as_copy_dest(list: *mut c_void, res: usize, op: impl FnOnce()) {
    transition(list, res, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    op();
    transition(list, res, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
}

unsafe fn copy_buffer_region(list: *mut c_void, dst: usize, dst_off: u64, src: usize, src_off: u64, n: u64) {
    pc::<unsafe extern "system" fn(*mut c_void, usize, u64, usize, u64, u64)>(list, PC_COPY_BUFFER_REGION)(list, dst, dst_off, src, src_off, n);
}

fn warn(counter: &AtomicU64, what: &str, why: std::fmt::Arguments) {
    if once(counter, 4) {
        crate::xlog_warn!("D3D12.X: {what}: {why} – not done");
    }
}

// ---------------------------------------------------------------------------------------
// per-list state & live queue tracking (KickoffX, Wait32/64BitValueX, RSSetDepthBiasX,
// index buffer restoration, nested occlusion queries)
// ---------------------------------------------------------------------------------------

/// a query: (heap, index)
type Query = (usize, u32);

/// One list's occlusion queries: the one active on the PC (with its type) and the outer ones
/// it ended early.
#[derive(Default)]
struct Occlusion {
    active: Option<(Query, D3D12_QUERY_TYPE)>,
    ended_early: Vec<Query>,
}

impl Occlusion {
    /// Begin `q` on the PC; Some(outer): end that one on the PC first.
    fn begin(&mut self, q: Query, ty: D3D12_QUERY_TYPE) -> Option<(Query, D3D12_QUERY_TYPE)> {
        let outer = self.active.replace((q, ty));
        if let Some((o, _)) = outer {
            self.ended_early.push(o);
        }
        outer
    }

    /// End the currently active PC query early (e.g. before a `KickoffX` split), so `EndQuery`
    /// later resolves it as visible.
    fn end_active_early(&mut self) -> Option<(Query, D3D12_QUERY_TYPE)> {
        let active = self.active.take()?;
        self.ended_early.push(active.0);
        Some(active)
    }

    /// Whether the PC ends `q` (not: an outer one already ended – it resolves as visible).
    fn end(&mut self, q: Query) -> bool {
        if self.active.is_some_and(|(a, _)| a == q) {
            self.active = None;
            return true;
        }
        match self.ended_early.iter().position(|e| *e == q) {
            Some(i) => {
                self.ended_early.swap_remove(i);
                false
            }
            None => true,
        }
    }
}

/// Recorded GPU-side memory wait from `Wait32BitValueX` (#71) or `Wait64BitValueX` (#72).
/// Evaluates `(mem_value & mask) <comparison> value` using `D3D12_COMPARISON_FUNC` (1..=8),
/// matching AMD PM4 `WAIT_REG_MEM` semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingWait {
    va: u64,
    width: u8, // 4 or 8
    comparison: u32,
    value: u64,
    mask: u64,
}

fn compare_u64(op: u32, lhs: u64, rhs: u64) -> Option<bool> {
    match op {
        1 => Some(false),        // D3D12_COMPARISON_FUNC_NEVER
        2 => Some(lhs < rhs),    // D3D12_COMPARISON_FUNC_LESS
        3 => Some(lhs == rhs),   // D3D12_COMPARISON_FUNC_EQUAL
        4 => Some(lhs <= rhs),   // D3D12_COMPARISON_FUNC_LESS_EQUAL
        5 => Some(lhs > rhs),    // D3D12_COMPARISON_FUNC_GREATER
        6 => Some(lhs != rhs),   // D3D12_COMPARISON_FUNC_NOT_EQUAL
        7 => Some(lhs >= rhs),   // D3D12_COMPARISON_FUNC_GREATER_EQUAL
        8 => Some(true),         // D3D12_COMPARISON_FUNC_ALWAYS
        _ => None,
    }
}

impl PendingWait {
    unsafe fn read_value(&self) -> Option<u64> {
        let addr = self.va as usize;
        if !crate::gpu::shader::readable(addr, self.width as usize) {
            return None;
        }
        match self.width {
            4 => Some((*(addr as *const AtomicU32)).load(Ordering::Acquire) as u64),
            8 => Some((*(addr as *const AtomicU64)).load(Ordering::Acquire)),
            _ => None,
        }
    }

    unsafe fn is_satisfied(&self) -> bool {
        let Some(cur) = self.read_value() else {
            return true;
        };
        compare_u64(self.comparison, cur & self.mask, self.value).unwrap_or(true)
    }

    unsafe fn wait_until_satisfied(&self, timeout: Duration) -> bool {
        if self.is_satisfied() {
            return true;
        }
        for _ in 0..512 {
            std::hint::spin_loop();
            if self.is_satisfied() {
                return true;
            }
        }
        let start = Instant::now();
        while start.elapsed() < timeout {
            let step = if start.elapsed() < Duration::from_millis(10) {
                Duration::from_micros(100)
            } else {
                Duration::from_millis(1)
            };
            std::thread::sleep(step);
            if self.is_satisfied() {
                return true;
            }
        }
        self.is_satisfied()
    }
}

#[derive(Default)]
struct ListState {
    allocator: usize,
    list_type: Option<D3D12_COMMAND_LIST_TYPE>,
    last_queue: Option<usize>,
    /// Title's currently bound PSO (before dynamic depth-bias override)
    bound_pso: usize,
    /// Active `RSSetDepthBiasX` override (None = zero / disabled)
    depth_bias: Option<DepthBias>,
    /// Last depth bias sent to `ID3D12GraphicsCommandList9::RSSetDepthBias` on this PC list
    /// since CreateCommandList / Reset / ClearState / KickoffX (resets to ZERO on each).
    native_bias: DepthBias,
    /// True when the PC list's bound PSO needs patching via the fallback cache before the next draw
    pso_dirty: bool,
    /// True when a fallback depth-bias derived PSO is currently bound on the PC list
    using_fallback_pso: bool,
    /// Bound index buffer (PC address), restored after `DrawIndexedX` and `KickoffX`
    index_buffer: Option<D3D12_INDEX_BUFFER_VIEW>,
    /// Last `OMSetDepthBounds` values, restored after `KickoffX`
    depth_bounds: Option<(f32, f32)>,
    occlusion: Occlusion,
    /// UMA address ranges written by this list since last Reset/KickoffX
    written_ranges: Vec<(u64, u64)>,
    /// External memory waits recorded on this list
    pending_waits: Vec<PendingWait>,
}

impl ListState {
    fn note_write(&mut self, va: u64, size: u64) {
        if size == 0 {
            return;
        }
        let end = va.saturating_add(size);
        if self.written_ranges.len() < MAX_WRITTEN_RANGES {
            self.written_ranges.push((va, end));
        } else if let Some(last) = self.written_ranges.last_mut() {
            last.0 = last.0.min(va);
            last.1 = last.1.max(end);
        }
    }

    fn has_prior_write(&self, va: u64, size: u64) -> bool {
        let end = va.saturating_add(size);
        self.written_ranges.iter().any(|&(s, e)| va < e && s < end)
    }

    fn clear_waits(&mut self) {
        if !self.pending_waits.is_empty() {
            self.pending_waits.clear();
            PENDING_WAIT_LISTS.fetch_sub(1, Ordering::AcqRel);
        }
    }

    fn push_wait(&mut self, w: PendingWait) {
        if self.pending_waits.is_empty() {
            PENDING_WAIT_LISTS.fetch_add(1, Ordering::AcqRel);
        }
        self.pending_waits.push(w);
    }
}

#[derive(Clone, Copy)]
struct QueueInfo {
    queue: usize,
    device: usize,
    ty: D3D12_COMMAND_LIST_TYPE,
}

static LISTS: Mutex<Option<HashMap<usize, ListState>>> = Mutex::new(None);
static LIVE_QUEUES: Mutex<Vec<QueueInfo>> = Mutex::new(Vec::new());
/// Number of lists with at least one pending `Wait32/64BitValueX` (fast-path guard for ExecuteCommandLists).
static PENDING_WAIT_LISTS: AtomicUsize = AtomicUsize::new(0);
/// Set once any list uses `RSSetDepthBiasX` (fast-path guard for Draw* hooks).
static DEPTH_BIAS_USED: AtomicBool = AtomicBool::new(false);

/// Record a newly created title command queue so `KickoffX` can find a compatible queue.
pub(super) fn on_queue_create(queue: usize, device: usize, ty: D3D12_COMMAND_LIST_TYPE) {
    if queue == 0 {
        return;
    }
    let mut g = LIVE_QUEUES.lock().unwrap();
    g.retain(|q| q.queue != queue);
    g.push(QueueInfo { queue, device, ty });
}

/// Remove a released command queue so `KickoffX` never submits to a dead queue.
pub(super) fn on_queue_release(queue: usize) {
    LIVE_QUEUES.lock().unwrap().retain(|q| q.queue != queue);
    if let Some(m) = LISTS.lock().unwrap().as_mut() {
        for s in m.values_mut() {
            if s.last_queue == Some(queue) {
                s.last_queue = None;
            }
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Optional test interceptor for `try_native_depth_bias` so unit tests deterministically
    /// verify the native `ID3D12GraphicsCommandList9::RSSetDepthBias` path and its fallback
    /// even on Windows builds whose inbox D3D12Core.dll stops at `ID3D12GraphicsCommandList7`.
    static TEST_NATIVE_BIAS: std::cell::RefCell<Option<Box<dyn FnMut(usize, DepthBias) -> bool>>> = const { std::cell::RefCell::new(None) };
}

/// Apply depth bias natively via `ID3D12GraphicsCommandList9::RSSetDepthBias` (exact f32,
/// without rounding to i32 or deriving a patched PSO). Returns `false` if unavailable.
unsafe fn try_native_depth_bias(list: *mut c_void, bias: DepthBias) -> bool {
    if list.is_null() {
        return false;
    }
    #[cfg(test)]
    if let Some(res) = TEST_NATIVE_BIAS.with(|h| h.borrow_mut().as_mut().map(|f| f(list as usize, bias))) {
        return res;
    }
    let Some(cl) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) else {
        return false;
    };
    let Ok(cl9) = cl.cast::<ID3D12GraphicsCommandList9>() else {
        return false;
    };
    cl9.RSSetDepthBias(bias.depth_bias(), bias.depth_bias_clamp(), bias.slope_scaled_depth_bias());
    true
}

/// Record a newly created command list's allocator, initial pipeline, and type.
pub(super) unsafe fn on_create(list: *mut c_void, alloc: usize, pso: usize, ty: D3D12_COMMAND_LIST_TYPE) {
    if list.is_null() {
        return;
    }
    let key = list as usize;
    {
        let mut g = LISTS.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        if let Some(old) = m.remove(&key) {
            if !old.pending_waits.is_empty() {
                PENDING_WAIT_LISTS.fetch_sub(1, Ordering::AcqRel);
            }
        }
        m.insert(key, ListState { allocator: alloc, list_type: Some(ty), bound_pso: pso, ..Default::default() });
    }
    // Dynamic-depth-bias PSOs ignore baked RasterizerState depth bias; seed the list with the
    // initial PSO's static bias if non-zero (CreateCommandList resets RSSetDepthBias to 0).
    if ty != D3D12_COMMAND_LIST_TYPE_BUNDLE && pso != 0 {
        if let Some((true, static_bias)) = device_x::base_depth_bias_info(pso) {
            if !static_bias.is_zero() && try_native_depth_bias(list, static_bias) {
                if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                    s.native_bias = static_bias;
                }
            }
        }
    }
}

/// Clean up all tracked state when a command list is released for good.
pub(super) fn on_release(list: usize) {
    if let Some(old) = LISTS.lock().unwrap().as_mut().and_then(|m| m.remove(&list)) {
        if !old.pending_waits.is_empty() {
            PENDING_WAIT_LISTS.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// Forget a released command allocator so `KickoffX` will not reset a list with a dead allocator.
pub(super) fn on_allocator_release(alloc: usize) {
    if let Some(m) = LISTS.lock().unwrap().as_mut() {
        for s in m.values_mut() {
            if s.allocator == alloc {
                s.allocator = 0;
            }
        }
    }
}

/// Track `SetPipelineState` on a command list (for `RSSetDepthBiasX` and `KickoffX`),
/// syncing `ID3D12GraphicsCommandList9::RSSetDepthBias` when the bound PSO uses native dynamic
/// depth bias.
pub(super) unsafe fn on_set_pipeline_state(list: *mut c_void, pso: usize) {
    let key = list as usize;
    let sync_target = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(key).or_default();
        // Keep the title-visible base PSO key when `bind_abi` re-binds its resolved background PSO.
        if s.bound_pso == 0 || pso == s.bound_pso || super::pso_async::resolve(s.bound_pso) != pso {
            s.bound_pso = pso;
        }
        let effective = s.bound_pso;
        s.using_fallback_pso = false;
        let is_bundle = s.list_type == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE);
        let info = if effective != 0 { device_x::base_depth_bias_info(effective) } else { None };
        if !is_bundle {
            if let Some((true, static_bias)) = info {
                let target = s.depth_bias.unwrap_or(static_bias);
                s.pso_dirty = false;
                if target != s.native_bias {
                    Some((target, s.depth_bias.is_some()))
                } else {
                    None
                }
            } else {
                s.pso_dirty = s.depth_bias.is_some();
                if s.pso_dirty {
                    DEPTH_BIAS_USED.store(true, Ordering::Release);
                }
                None
            }
        } else {
            s.pso_dirty = s.depth_bias.is_some();
            if s.pso_dirty {
                DEPTH_BIAS_USED.store(true, Ordering::Release);
            }
            None
        }
    };
    if let Some((target, has_override)) = sync_target {
        if try_native_depth_bias(list, target) {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                s.native_bias = target;
            }
        } else if has_override {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                s.pso_dirty = true;
            }
            DEPTH_BIAS_USED.store(true, Ordering::Release);
        }
    }
}

/// Track `ClearState` on a command list: resets dynamic state and binds `pso`.
pub(super) unsafe fn on_clear_state(list: *mut c_void, pso: usize) {
    let key = list as usize;
    let is_bundle = {
        let mut g = LISTS.lock().unwrap();
        if let Some(s) = g.get_or_insert_with(HashMap::new).get_mut(&key) {
            s.bound_pso = pso;
            s.depth_bias = None;
            s.native_bias = DepthBias::ZERO;
            s.pso_dirty = false;
            s.using_fallback_pso = false;
            s.index_buffer = None;
            s.depth_bounds = None;
            s.list_type == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE)
        } else {
            false
        }
    };
    if !is_bundle && pso != 0 {
        if let Some((true, static_bias)) = device_x::base_depth_bias_info(pso) {
            if !static_bias.is_zero() && try_native_depth_bias(list, static_bias) {
                if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                    s.native_bias = static_bias;
                }
            }
        }
    }
}

/// Associate executed lists with `queue` and keep `queue` at the head of `LIVE_QUEUES`.
pub(super) unsafe fn note_queue_for_lists(queue: usize, lists: &[*mut c_void]) {
    if queue == 0 {
        return;
    }
    {
        let mut g = LIVE_QUEUES.lock().unwrap();
        if let Some(pos) = g.iter().position(|q| q.queue == queue) {
            let info = g.remove(pos);
            g.push(info);
        } else {
            let raw = queue as *mut c_void;
            if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&raw) {
                let ty = q.GetDesc().Type;
                let mut dev: Option<ID3D12Device> = None;
                let dev_raw = if q.GetDevice(&mut dev).is_ok() {
                    dev.as_ref().map_or(0, |d| d.as_raw() as usize)
                } else {
                    0
                };
                g.push(QueueInfo { queue, device: dev_raw, ty });
            }
        }
    }
    let mut g = LISTS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    for &l in lists {
        if !l.is_null() {
            m.entry(l as usize).or_default().last_queue = Some(queue);
        }
    }
}

fn pick_queue(last_queue: Option<usize>, device: usize, ty: D3D12_COMMAND_LIST_TYPE) -> Option<usize> {
    let g = LIVE_QUEUES.lock().unwrap();
    if let Some(q) = last_queue {
        if g.iter().any(|info| info.queue == q) {
            return Some(q);
        }
    }
    g.iter().rev().find(|info| info.ty == ty && (device == 0 || info.device == 0 || info.device == device)).map(|info| info.queue)
}

fn note_write(list: usize, va: u64, size: u64) {
    LISTS.lock().unwrap().get_or_insert_with(HashMap::new).entry(list).or_default().note_write(va, size);
}

unsafe fn wait_all(waits: &[PendingWait]) {
    static TIMEOUTS: AtomicU64 = AtomicU64::new(0);
    for w in waits {
        if !w.wait_until_satisfied(WAIT_TIMEOUT) && once(&TIMEOUTS, 4) {
            crate::xlog_warn!(
                "D3D12.X: Wait{}BitValueX timed out after {} ms at {:#x} (op {}, want {:#x}, mask {:#x})",
                w.width * 8,
                WAIT_TIMEOUT.as_millis(),
                w.va,
                w.comparison,
                w.value,
                w.mask
            );
        }
    }
}

/// Submit `lists` through `submit`, splitting the batch before any list whose `Wait32/64BitValueX`
/// condition is not yet satisfied so earlier lists in the same batch can run on the GPU first.
pub(super) unsafe fn execute_with_waits(lists: &[*mut c_void], mut submit: impl FnMut(&[*mut c_void])) {
    if lists.is_empty() || PENDING_WAIT_LISTS.load(Ordering::Acquire) == 0 {
        submit(lists);
        return;
    }
    let per_list: Vec<Vec<PendingWait>> = {
        let g = LISTS.lock().unwrap();
        lists
            .iter()
            .map(|&l| {
                g.as_ref()
                    .and_then(|m| m.get(&(l as usize)))
                    .map(|s| s.pending_waits.clone())
                    .unwrap_or_default()
            })
            .collect()
    };
    let mut start = 0usize;
    for (i, waits) in per_list.iter().enumerate() {
        if waits.is_empty() || waits.iter().all(|w| w.is_satisfied()) {
            continue;
        }
        if i > start {
            submit(&lists[start..i]);
        }
        wait_all(waits);
        start = i;
    }
    if start < lists.len() {
        submit(&lists[start..]);
    }
}

// ---------------------------------------------------------------------------------------
// memory copy / fill / immediate write / timestamp
// ---------------------------------------------------------------------------------------

/// command list #60 CopyMemoryToMemoryX(destination, source, size, flags)
pub(super) unsafe extern "system" fn copy_memory_x(this: *mut c_void, dst: u64, src: u64, size: u64, _flags: u32) {
    static N: AtomicU64 = AtomicU64::new(0);
    const W: &str = "CopyMemoryToMemoryX";
    if size == 0 || !size.is_multiple_of(4) {
        return warn(&N, W, format_args!("size {size:#x} is not a non-zero multiple of 4"));
    }
    let (Some((d, doff)), Some((s, soff))) = (real::uma_window_buffer(dst as usize, size as usize), real::uma_window_buffer(src as usize, size as usize)) else {
        return warn(&N, W, format_args!("{src:#x} → {dst:#x} ({size:#x} bytes) is not in unified memory"));
    };
    if d == s && dst < src + size && src < dst + size {
        return warn(&N, W, format_args!("{src:#x} → {dst:#x} ({size:#x} bytes) overlap"));
    }
    note_write(this as usize, dst, size);
    if d == s {
        // one window: the buffer is the source and the destination (the ranges do not overlap)
        as_copy_dest(this, d, || copy_buffer_region(this, d, doff, s, soff, size));
        return;
    }
    transition(this, s, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    as_copy_dest(this, d, || copy_buffer_region(this, d, doff, s, soff, size));
    transition(this, s, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
}

unsafe fn device_of(list: *mut c_void) -> Option<ID3D12Device> {
    let mut dev: Option<ID3D12Device> = None;
    ID3D12GraphicsCommandList::from_raw_borrowed(&list)?.GetDevice(&mut dev).ok()?;
    dev
}

/// upload buffers holding a repeated 16-byte pattern: (pattern, buffer, bytes)
static PATTERNS: Mutex<Vec<([u8; 16], ID3D12Resource, u64)>> = Mutex::new(Vec::new());

/// A buffer of at least `n` bytes repeating `pat` (the device's upload heap).
unsafe fn pattern_buffer(list: *mut c_void, pat: [u8; 16], n: u64) -> Option<usize> {
    let mut cache = PATTERNS.lock().unwrap();
    if let Some((_, b, _)) = cache.iter().find(|(p, _, len)| *p == pat && *len >= n) {
        return Some(b.as_raw() as usize);
    }
    // a larger one replaces an older pattern buffer only when nothing may still read it: kept
    if cache.len() >= 32 {
        return None;
    }
    let dev = device_of(list)?;
    let len = n.next_power_of_two().max(0x1_0000);
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: len,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut b: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        None,
        &mut b,
    )
    .ok()?;
    let b = b?;
    let mut p: *mut c_void = std::ptr::null_mut();
    b.Map(0, None, Some(&mut p)).ok()?;
    for chunk in std::slice::from_raw_parts_mut(p as *mut u8, len as usize).chunks_exact_mut(16) {
        chunk.copy_from_slice(&pat);
    }
    b.Unmap(0, None);
    let raw = b.as_raw() as usize;
    cache.push((pat, b, len));
    Some(raw)
}

unsafe fn fill(list: *mut c_void, va: u64, size: u64, pat: [u8; 16], unit: u64, what: &str) {
    static N: AtomicU64 = AtomicU64::new(0);
    if size == 0 || !size.is_multiple_of(unit) || size > MAX_FILL {
        return warn(&N, what, format_args!("size {size:#x} (a non-zero multiple of {unit} up to {MAX_FILL:#x} is filled)"));
    }
    let Some((d, off)) = real::uma_window_buffer(va as usize, size as usize) else {
        return warn(&N, what, format_args!("{va:#x} ({size:#x} bytes) is not in unified memory"));
    };
    let Some(src) = pattern_buffer(list, pat, size) else {
        return warn(&N, what, format_args!("no pattern buffer for {pat:x?}"));
    };
    note_write(list as usize, va, size);
    as_copy_dest(list, d, || copy_buffer_region(list, d, off, src, 0, size));
}

fn pattern(bytes: &[u8]) -> [u8; 16] {
    let mut p = [0u8; 16];
    for (i, b) in p.iter_mut().enumerate() {
        *b = bytes[i % bytes.len()];
    }
    p
}

/// command list #61 FillMemoryWith32BitValueX(address, size, value)
pub(super) unsafe extern "system" fn fill32_x(this: *mut c_void, va: u64, size: u64, value: u32) {
    fill(this, va, size, pattern(&value.to_le_bytes()), 4, "FillMemoryWith32BitValueX");
}

/// command list #62 FillMemoryWith64BitValueX(address, size, value)
pub(super) unsafe extern "system" fn fill64_x(this: *mut c_void, va: u64, size: u64, value: u64) {
    fill(this, va, size, pattern(&value.to_le_bytes()), 8, "FillMemoryWith64BitValueX");
}

/// command list #63 FillMemoryWith128BitValueX(address, size, value*)
pub(super) unsafe extern "system" fn fill128_x(this: *mut c_void, va: u64, size: u64, value: *const [u8; 16]) {
    if !value.is_null() {
        fill(this, va, size, value.read_unaligned(), 16, "FillMemoryWith128BitValueX");
    }
}

/// Values written by the GPU at `va` (WriteBufferImmediate: MARKER_IN = when the command is
/// reached, MARKER_OUT = when the work before it is done).
unsafe fn write(list: *mut c_void, va: u64, words: &[u32], mode: D3D12_WRITEBUFFERIMMEDIATE_MODE, what: &str) {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = 4 * words.len() as u64;
    if !va.is_multiple_of(n) {
        return warn(&N, what, format_args!("{va:#x} is not {n}-byte aligned"));
    }
    let (Some((d, _)), Some(at)) = (real::uma_window_buffer(va as usize, n as usize), real::uma_va_of(va as usize)) else {
        return warn(&N, what, format_args!("{va:#x} is not in unified memory"));
    };
    note_write(list as usize, va, n);
    let params: Vec<D3D12_WRITEBUFFERIMMEDIATE_PARAMETER> = words.iter().enumerate().map(|(i, &w)| D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: at + 4 * i as u64, Value: w }).collect();
    let modes = vec![mode; words.len()];
    type F = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER, *const D3D12_WRITEBUFFERIMMEDIATE_MODE);
    as_copy_dest(list, d, || pc::<F>(list, PC_WRITE_BUFFER_IMMEDIATE)(list, params.len() as u32, params.as_ptr(), modes.as_ptr()));
}

/// timestamp queries (one heap per device, used round-robin)
static TIMESTAMPS: Mutex<Option<HashMap<usize, ID3D12QueryHeap>>> = Mutex::new(None);
const TIMESTAMP_SLOTS: u32 = 4096;
static NEXT_TIMESTAMP: AtomicU32 = AtomicU32::new(0);

/// The GPU timestamp (the PC queue's ticks) written at `va` when the work before it is done.
unsafe fn timestamp(list: *mut c_void, va: u64, what: &str) {
    static N: AtomicU64 = AtomicU64::new(0);
    let ty = list_type(list);
    if ty != D3D12_COMMAND_LIST_TYPE_DIRECT && ty != D3D12_COMMAND_LIST_TYPE_COMPUTE {
        return warn(&N, what, format_args!("a timestamp on a type-{} list", ty.0));
    }
    let Some((d, off)) = real::uma_window_buffer(va as usize, 8).filter(|_| va.is_multiple_of(8)) else {
        return warn(&N, what, format_args!("{va:#x} is not 8-byte aligned in unified memory"));
    };
    let Some(dev) = device_of(list) else { return };
    let heap = {
        let mut m = TIMESTAMPS.lock().unwrap();
        let m = m.get_or_insert_with(HashMap::new);
        if let std::collections::hash_map::Entry::Vacant(slot) = m.entry(dev.as_raw() as usize) {
            let mut h: Option<ID3D12QueryHeap> = None;
            let desc = D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: TIMESTAMP_SLOTS, NodeMask: 0 };
            match dev.CreateQueryHeap(&desc, &mut h) {
                Ok(()) if h.is_some() => slot.insert(h.unwrap()),
                _ => return warn(&N, what, format_args!("no timestamp query heap")),
            };
        }
        m[&(dev.as_raw() as usize)].as_raw() as usize
    };
    note_write(list as usize, va, 8);
    let i = NEXT_TIMESTAMP.fetch_add(1, Ordering::Relaxed) % TIMESTAMP_SLOTS;
    pc::<unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32)>(list, PC_END_QUERY)(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, i);
    type Resolve = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32, u32, usize, u64);
    as_copy_dest(list, d, || pc::<Resolve>(list, PC_RESOLVE_QUERY_DATA)(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, i, 1, d, off));
}

/// command list #69 Write32BitValueBottomOfPipeX(address, value, flush, flags)
pub(super) unsafe extern "system" fn write32_bottom_x(this: *mut c_void, va: u64, value: u32, _flush: u32, _flags: u32) {
    write(this, va, &[value], D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT, "Write32BitValueBottomOfPipeX");
}

/// command list #70 Write64BitValueBottomOfPipeX(address, value, flush, flags)
pub(super) unsafe extern "system" fn write64_bottom_x(this: *mut c_void, va: u64, value: u64, _flush: u32, flags: u32) {
    const W: &str = "Write64BitValueBottomOfPipeX";
    if flags & BOTTOM_TIMESTAMPS != 0 {
        return timestamp(this, va, W);
    }
    write(this, va, &[value as u32, (value >> 32) as u32], D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT, W);
}

/// Record a `Wait32BitValueX` or `Wait64BitValueX` memory wait on `list`.
///
/// If an earlier write in the same open list already targeted an overlapping UMA range, the
/// `COMMON -> COPY_DEST -> COMMON` barrier around that write already orders subsequent work
/// on the GPU, so we do not stall before submission (which would deadlock waiting on our own
/// unsubmitted list). Otherwise the wait is evaluated on the CPU before `ExecuteCommandLists`
/// (or `KickoffX`) submits the list.
unsafe fn wait_memory(list: *mut c_void, va: u64, width: u8, comparison: u32, value: u64, mask: u64, what: &str) {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = width as u64;
    if !va.is_multiple_of(n) {
        return warn(&N, what, format_args!("{va:#x} is not {n}-byte aligned"));
    }
    if real::uma_window_buffer(va as usize, width as usize).is_none() {
        return warn(&N, what, format_args!("{va:#x} is not in unified memory"));
    }
    if compare_u64(comparison, 0, 0).is_none() {
        return warn(&N, what, format_args!("unknown comparison op {comparison}"));
    }
    if comparison == D3D12_COMPARISON_FUNC_ALWAYS.0 as u32 {
        return;
    }
    if comparison == D3D12_COMPARISON_FUNC_NEVER.0 as u32 {
        return warn(&N, what, format_args!("NEVER comparison at {va:#x} (skipped to avoid hang)"));
    }
    let mut g = LISTS.lock().unwrap();
    let s = g.get_or_insert_with(HashMap::new).entry(list as usize).or_default();
    if s.has_prior_write(va, n) {
        return;
    }
    s.push_wait(PendingWait { va, width, comparison, value, mask });
}

/// command list #71 Wait32BitValueX(address, comparison, value, mask, flags)
pub(super) unsafe extern "system" fn wait32_x(this: *mut c_void, va: u64, comparison: u32, value: u32, mask: u32, _flags: u32) {
    wait_memory(this, va, 4, comparison, value as u64, mask as u64, "Wait32BitValueX");
}

/// command list #72 Wait64BitValueX(address, comparison, value, mask, flags)
pub(super) unsafe extern "system" fn wait64_x(this: *mut c_void, va: u64, comparison: u32, value: u64, mask: u64, _flags: u32) {
    wait_memory(this, va, 8, comparison, value, mask, "Wait64BitValueX");
}

/// command list #88 Write32BitValueTopOfPipeX(address, value, flags)
pub(super) unsafe extern "system" fn write32_top_x(this: *mut c_void, va: u64, value: u32, _flags: u32) {
    write(this, va, &[value], D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN, "Write32BitValueTopOfPipeX");
}

/// command list #89 Write64BitValueTopOfPipeX(address, value, flags)
pub(super) unsafe extern "system" fn write64_top_x(this: *mut c_void, va: u64, value: u64, flags: u32) {
    const W: &str = "Write64BitValueTopOfPipeX";
    if flags & TOP_TIMESTAMPS != 0 {
        return timestamp(this, va, W);
    }
    write(this, va, &[value as u32, (value >> 32) as u32], D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN, W);
}

/// command list #90 Write32BitValueBottomOfShaderX(address, value, stage)
pub(super) unsafe extern "system" fn write32_bottom_of_shader_x(this: *mut c_void, va: u64, value: u32, _stage: u32) {
    write(this, va, &[value], D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT, "Write32BitValueBottomOfShaderX");
}

/// command list #68 OMSetDepthBounds(min, max) (PC #62)
pub(super) unsafe extern "system" fn set_depth_bounds_x(this: *mut c_void, min: f32, max: f32) {
    LISTS.lock().unwrap().get_or_insert_with(HashMap::new).entry(this as usize).or_default().depth_bounds = Some((min, max));
    pc::<unsafe extern "system" fn(*mut c_void, f32, f32)>(this, PC_OM_SET_DEPTH_BOUNDS)(this, min, max);
}

/// command list #78 DispatchX(x, y, z, flags)
pub(super) unsafe extern "system" fn dispatch_x(this: *mut c_void, x: u32, y: u32, z: u32, _flags: u32) {
    super::bind::dispatch(this, x, y, z);
}

/// command list #84 CopyResourceX(destination, source, flags); in place (the console's
/// compression operations) there is nothing to do on PC
pub(super) unsafe extern "system" fn copy_resource_flags_x(this: *mut c_void, dst: usize, src: usize, _flags: u32) {
    if dst != src {
        copy_resource_x(this, dst, src);
    }
}

/// command list #85 SetPredicationBufferX({ u64 address; u32 operation }*)
pub(super) unsafe extern "system" fn set_predication_buffer_x(this: *mut c_void, desc: *const u64) {
    static N: AtomicU64 = AtomicU64::new(0);
    type F = unsafe extern "system" fn(*mut c_void, usize, u64, D3D12_PREDICATION_OP);
    let f = pc::<F>(this, PC_SET_PREDICATION);
    if desc.is_null() {
        return f(this, 0, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
    }
    let (va, op) = (desc.read_unaligned(), (desc.add(1) as *const u32).read_unaligned());
    let Some((b, off)) = real::uma_window_buffer(va as usize, 8).filter(|_| va.is_multiple_of(8)) else {
        return warn(&N, "SetPredicationBufferX", format_args!("{va:#x} is not 8-byte aligned in unified memory"));
    };
    transition(this, b, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_PREDICATION);
    f(this, b, off, D3D12_PREDICATION_OP(op as i32));
    transition(this, b, D3D12_RESOURCE_STATE_PREDICATION, D3D12_RESOURCE_STATE_COMMON);
}

/// command list #93 KickoffX(): flush commands recorded so far by closing the list, submitting
/// it to a compatible live command queue, and resetting it on the same command allocator so
/// recording can continue seamlessly until the title's normal `Close` + `ExecuteCommandLists`.
///
/// If no compatible queue or allocator is known yet (e.g. a bundle, or a list created before
/// any queue of its type exists), `KickoffX` is a safe logged no-op and the recorded work runs
/// at `Close` + `ExecuteCommandLists`.
pub(super) unsafe extern "system" fn kickoff_x(this: *mut c_void) {
    static NO_QUEUE: AtomicU64 = AtomicU64::new(0);
    static ERR: AtomicU64 = AtomicU64::new(0);
    let ty = list_type(this);
    if ty == D3D12_COMMAND_LIST_TYPE_BUNDLE {
        return;
    }
    let dev_raw = device_of(this).map_or(0, |d| d.as_raw() as usize);
    let (alloc, queue, bound_pso, depth_bias, index_buffer, depth_bounds, active_query, waits) = {
        let mut g = LISTS.lock().unwrap();
        let Some(s) = g.as_mut().and_then(|m| m.get_mut(&(this as usize))) else {
            if once(&NO_QUEUE, 4) {
                crate::xlog_info!("D3D12.X: KickoffX – list allocator unknown, deferring to Close + ExecuteCommandLists");
            }
            return;
        };
        if s.allocator == 0 {
            if once(&NO_QUEUE, 4) {
                crate::xlog_info!("D3D12.X: KickoffX – list allocator unknown, deferring to Close + ExecuteCommandLists");
            }
            return;
        }
        let Some(q) = pick_queue(s.last_queue, dev_raw, s.list_type.unwrap_or(ty)) else {
            if once(&NO_QUEUE, 4) {
                crate::xlog_info!("D3D12.X: KickoffX – no live queue of type {} yet, deferring to Close + ExecuteCommandLists", ty.0);
            }
            return;
        };
        let active_query = s.occlusion.end_active_early();
        let waits = if s.pending_waits.is_empty() {
            Vec::new()
        } else {
            PENDING_WAIT_LISTS.fetch_sub(1, Ordering::AcqRel);
            std::mem::take(&mut s.pending_waits)
        };
        s.written_ranges.clear();
        (s.allocator, q, s.bound_pso, s.depth_bias, s.index_buffer, s.depth_bounds, active_query, waits)
    };

    // End any active occlusion query before Close (PC forbids closing with an open query).
    if let Some(((h, i), qty)) = active_query {
        pc::<QueryFn>(this, PC_END_QUERY)(this, h, qty, i);
    }

    let close_hr = pc::<unsafe extern "system" fn(*mut c_void) -> i32>(this, PC_CLOSE)(this);
    if close_hr < 0 {
        return warn(&ERR, "KickoffX", format_args!("Close failed ({close_hr:#x})"));
    }

    let q_raw = queue as *mut c_void;
    super::title_textures::refresh(q_raw);
    if !waits.is_empty() {
        wait_all(&waits);
    }

    type ExecFn = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
    let exec: ExecFn = transmute_copy(&real::orig_fn(q_raw, PC_QUEUE_EXECUTE_COMMAND_LISTS));
    let one = [this];
    exec(q_raw, 1, one.as_ptr());

    let init_pso = super::pso_async::resolve_for_bind(bound_pso);
    let ext = *(this as *const usize);
    let reset_hr = pc::<unsafe extern "system" fn(*mut c_void, usize, usize) -> i32>(this, PC_RESET)(this, alloc, init_pso);
    if *(this as *const usize) != ext {
        *(this as *mut usize) = ext;
    }
    if reset_hr < 0 {
        return warn(&ERR, "KickoffX", format_args!("Reset after kickoff failed ({reset_hr:#x})"));
    }

    // Reset() cleared the PC list's native RSSetDepthBias state to (0, 0, 0) and bound init_pso.
    if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(this as usize))) {
        s.native_bias = DepthBias::ZERO;
        s.using_fallback_pso = false;
        s.pso_dirty = false;
    }
    let info = if bound_pso != 0 { device_x::base_depth_bias_info(bound_pso) } else { None };
    let mut applied_native = false;
    if let Some((true, static_bias)) = info {
        let target = depth_bias.unwrap_or(static_bias);
        if target.is_zero() {
            applied_native = true;
        } else if try_native_depth_bias(this, target) {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(this as usize))) {
                s.native_bias = target;
            }
            applied_native = true;
        }
    } else if bound_pso == 0 {
        if let Some(b) = depth_bias {
            if device_x::dynamic_depth_bias_supported(dev_raw as *mut c_void) && try_native_depth_bias(this, b) {
                if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(this as usize))) {
                    s.native_bias = b;
                }
                applied_native = true;
            }
        }
    }
    if depth_bias.is_some() && !applied_native {
        if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(this as usize))) {
            s.pso_dirty = true;
        }
        DEPTH_BIAS_USED.store(true, Ordering::Release);
    }
    if let Some(ref view) = index_buffer {
        set_index_buffer(this, Some(view));
    }
    if let Some((min, max)) = depth_bounds {
        pc::<unsafe extern "system" fn(*mut c_void, f32, f32)>(this, PC_OM_SET_DEPTH_BOUNDS)(this, min, max);
    }
}

/// command list #94 RSSetDepthBiasX(depthBias, depthBiasClamp, slopeScaledDepthBias):
/// dynamic depth bias override. Uses `ID3D12GraphicsCommandList9::RSSetDepthBias` natively
/// when `D3D12_FEATURE_D3D12_OPTIONS16::DynamicDepthBiasSupported` and the bound PSO support
/// it (preserving exact `f32` bias with zero PSO derivation), falling back to the patched-PSO
/// cache (`device_x::depth_bias_pso`) otherwise.
pub(super) unsafe extern "system" fn set_depth_bias_x(this: *mut c_void, depth_bias: f32, depth_bias_clamp: f32, slope_scaled_depth_bias: f32) {
    static NON_FINITE: AtomicU64 = AtomicU64::new(0);
    let clean = |v: f32| {
        if v.is_finite() {
            v
        } else {
            if once(&NON_FINITE, 4) {
                crate::xlog_warn!("D3D12.X: RSSetDepthBiasX – non-finite value {v} clamped to 0.0");
            }
            0.0
        }
    };
    let bias = DepthBias::new(clean(depth_bias), clean(depth_bias_clamp), clean(slope_scaled_depth_bias));
    let key = this as usize;

    let (bound_pso, is_bundle) = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(key).or_default();
        (s.bound_pso, s.list_type == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE))
    };

    let can_native = if is_bundle {
        false
    } else if bound_pso != 0 {
        matches!(device_x::base_depth_bias_info(bound_pso), Some((true, _)))
    } else {
        device_of(this).is_some_and(|dev| device_x::dynamic_depth_bias_supported(dev.as_raw()))
    };

    if can_native {
        let target = if bias.is_zero() { DepthBias::ZERO } else { bias };
        let (need_restore_base, need_native_call) = {
            let mut g = LISTS.lock().unwrap();
            let s = g.get_or_insert_with(HashMap::new).entry(key).or_default();
            s.depth_bias = if bias.is_zero() { None } else { Some(bias) };
            s.pso_dirty = false;
            let restore = s.using_fallback_pso && s.bound_pso != 0;
            s.using_fallback_pso = false;
            (if restore { Some(s.bound_pso) } else { None }, target != s.native_bias)
        };
        if let Some(base) = need_restore_base {
            let resolved = super::pso_async::resolve_for_bind(base);
            pc::<unsafe extern "system" fn(*mut c_void, usize)>(this, PC_SET_PIPELINE_STATE)(this, resolved);
        }
        if !need_native_call {
            return;
        }
        if try_native_depth_bias(this, target) {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                s.native_bias = target;
            }
            return;
        }
        // ID3D12GraphicsCommandList9 unavailable: fall through to patched-PSO fallback.
        if !bias.is_zero() {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                s.pso_dirty = true;
            }
            DEPTH_BIAS_USED.store(true, Ordering::Release);
        }
        return;
    }

    let restore = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(key).or_default();
        if bias.is_zero() {
            let was_active = s.depth_bias.take().is_some();
            let was_fallback = std::mem::take(&mut s.using_fallback_pso);
            s.pso_dirty = false;
            if (was_active || was_fallback) && s.bound_pso != 0 {
                Some(s.bound_pso)
            } else {
                None
            }
        } else {
            if s.depth_bias != Some(bias) || !s.using_fallback_pso {
                s.depth_bias = Some(bias);
                s.pso_dirty = true;
                DEPTH_BIAS_USED.store(true, Ordering::Release);
            }
            None
        }
    };
    if let Some(base) = restore {
        let resolved = super::pso_async::resolve_for_bind(base);
        pc::<unsafe extern "system" fn(*mut c_void, usize)>(this, PC_SET_PIPELINE_STATE)(this, resolved);
    }
}

/// Apply a pending fallback `RSSetDepthBiasX` override before a draw call on `list`.
pub(super) unsafe fn prepare_draw(list: *mut c_void) {
    if !DEPTH_BIAS_USED.load(Ordering::Acquire) {
        return;
    }
    let (bound_pso, bias) = {
        let mut g = LISTS.lock().unwrap();
        let Some(s) = g.as_mut().and_then(|m| m.get_mut(&(list as usize))) else {
            return;
        };
        if !s.pso_dirty {
            return;
        }
        s.pso_dirty = false;
        (s.bound_pso, s.depth_bias)
    };
    if bound_pso == 0 {
        return;
    }
    let (target, is_fallback) = match bias {
        Some(b) => {
            static MISS: AtomicU64 = AtomicU64::new(0);
            let Some(dev) = device_of(list) else { return };
            match device_x::depth_bias_pso(dev.as_raw(), bound_pso, b) {
                Some(derived) => (derived, true),
                None => {
                    if once(&MISS, 4) {
                        crate::xlog_warn!("D3D12.X: RSSetDepthBiasX – could not derive pipeline for {bound_pso:#x}, using base pipeline");
                    }
                    return;
                }
            }
        }
        None => (bound_pso, false),
    };
    let resolved = super::pso_async::resolve_for_bind(target);
    pc::<unsafe extern "system" fn(*mut c_void, usize)>(list, PC_SET_PIPELINE_STATE)(list, resolved);
    if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(list as usize))) {
        s.using_fallback_pso = is_fallback;
    }
}

// ---------------------------------------------------------------------------------------
// index buffers & DrawIndexedX
// ---------------------------------------------------------------------------------------

/// a console GPU address as the PC's (unified memory: the window buffer's)
fn pc_va(va: u64) -> u64 {
    real::uma_va_of(va as usize).unwrap_or(va)
}

unsafe fn set_index_buffer(list: *mut c_void, view: Option<&D3D12_INDEX_BUFFER_VIEW>) {
    let f = pc::<unsafe extern "system" fn(*mut c_void, *const D3D12_INDEX_BUFFER_VIEW)>(list, PC_IA_SET_INDEX_BUFFER);
    f(list, view.map_or(std::ptr::null(), |v| v as *const _));
}

const PC_IA_SET_VERTEX_BUFFERS: usize = 44;

/// command list #48 IASetVertexBuffers(start slot, count, views) (PC #44): the addresses are
/// the title's (its memory: the GPU address of that memory for the PC)
pub(super) unsafe extern "system" fn ia_set_vertex_buffers_x(this: *mut c_void, start: u32, n: u32, views: *const D3D12_VERTEX_BUFFER_VIEW) {
    let f = pc::<unsafe extern "system" fn(*mut c_void, u32, u32, *const D3D12_VERTEX_BUFFER_VIEW)>(this, PC_IA_SET_VERTEX_BUFFERS);
    if views.is_null() || n == 0 {
        return f(this, start, n, views);
    }
    let v: Vec<D3D12_VERTEX_BUFFER_VIEW> = (0..n as usize)
        .map(|k| {
            let mut v = views.add(k).read_unaligned();
            v.BufferLocation = pc_va(v.BufferLocation);
            v
        })
        .collect();
    f(this, start, n, v.as_ptr())
}

/// command list #47 IASetIndexBuffer(view) (PC #43): the address is the title's
pub(super) unsafe extern "system" fn ia_set_index_buffer_x(this: *mut c_void, view: *const D3D12_INDEX_BUFFER_VIEW) {
    let mut g = LISTS.lock().unwrap();
    let s = g.get_or_insert_with(HashMap::new).entry(this as usize).or_default();
    if view.is_null() {
        s.index_buffer = None;
        drop(g);
        return set_index_buffer(this, None);
    }
    let mut v = view.read_unaligned();
    v.BufferLocation = pc_va(v.BufferLocation);
    s.index_buffer = Some(v);
    drop(g);
    set_index_buffer(this, Some(&v));
}

/// command list #81 DrawIndexedX(index address, index format, index count, flags): one
/// instance of `count` indices read from that address; the bound index buffer stays
pub(super) unsafe extern "system" fn draw_indexed_x(this: *mut c_void, va: u64, format: DXGI_FORMAT, count: u32, _flags: u32) {
    static N: AtomicU64 = AtomicU64::new(0);
    let stride = match format {
        DXGI_FORMAT_R16_UINT => 2,
        DXGI_FORMAT_R32_UINT => 4,
        f => return warn(&N, "DrawIndexedX", format_args!("index format {}", f.0)),
    };
    if count == 0 {
        return;
    }
    prepare_draw(this);
    let view = D3D12_INDEX_BUFFER_VIEW { BufferLocation: pc_va(va), SizeInBytes: count.saturating_mul(stride), Format: format };
    set_index_buffer(this, Some(&view));
    pc::<unsafe extern "system" fn(*mut c_void, u32, u32, u32, i32, u32)>(this, PC_DRAW_INDEXED)(this, count, 1, 0, 0, 0);
    let bound = LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(this as usize)).and_then(|s| s.index_buffer));
    set_index_buffer(this, bound.as_ref());
}

// ---------------------------------------------------------------------------------------
// nested occlusion queries & Reset
// ---------------------------------------------------------------------------------------

/// outer queries ended early, and the result they resolve to
static VISIBLE: Mutex<Option<HashMap<Query, u64>>> = Mutex::new(None);
/// what such an OCCLUSION query reports (BINARY_OCCLUSION: 1)
const VISIBLE_SAMPLES: u64 = 0x1_0000;

fn is_occlusion(ty: D3D12_QUERY_TYPE) -> bool {
    ty == D3D12_QUERY_TYPE_OCCLUSION || ty == D3D12_QUERY_TYPE_BINARY_OCCLUSION
}

type QueryFn = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32);

/// command list #10 Reset(allocator, initial pipeline): updates allocator/pipeline and clears
/// per-recording state (queries, index buffer, depth bias, pending waits, written ranges)
pub(super) unsafe extern "system" fn reset_x(this: *mut c_void, alloc: usize, pso: usize) -> i32 {
    let is_bundle = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(this as usize).or_default();
        s.allocator = alloc;
        s.bound_pso = pso;
        s.depth_bias = None;
        s.native_bias = DepthBias::ZERO;
        s.pso_dirty = false;
        s.using_fallback_pso = false;
        s.index_buffer = None;
        s.depth_bounds = None;
        s.occlusion = Occlusion::default();
        s.written_ranges.clear();
        s.clear_waits();
        s.list_type == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE)
    };
    super::bind::reset(this, alloc);
    // the PC runtime writes its own vtable into a copy list again on Reset: keep XCL's
    let ext = *(this as *const usize);
    let resolved = super::pso_async::resolve_for_bind(pso);
    let hr = pc::<unsafe extern "system" fn(*mut c_void, usize, usize) -> i32>(this, PC_RESET)(this, alloc, resolved);
    if *(this as *const usize) != ext {
        *(this as *mut usize) = ext;
    }
    if hr >= 0 && !is_bundle && pso != 0 {
        if let Some((true, static_bias)) = device_x::base_depth_bias_info(pso) {
            if !static_bias.is_zero() && try_native_depth_bias(this, static_bias) {
                if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(this as usize))) {
                    s.native_bias = static_bias;
                }
            }
        }
    }
    hr
}

/// the D3D12_RESOURCE_STATES bits the PC knows
const PC_STATES: u32 = 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x20 | 0x40 | 0x80 | 0x100 | 0x200 | 0x400 | 0x800 | 0x1000 | 0x2000 | 0xF_0000 | 0x20_0000 | 0x40_0000 | 0x80_0000 | 0x100_0000;

/// #26 ResourceBarrier(count, barriers): console barriers (D3D12_RESOURCE_BARRIER, 32 bytes) as
/// PC ones – state bits the PC does not know are dropped (logged: the console's own), a
/// transition left with nothing to do goes, aliasing barriers cover any resources
pub(super) unsafe extern "system" fn resource_barrier_x(this: *mut c_void, n: u32, barriers: *const [u32; 8]) {
    static N: AtomicU64 = AtomicU64::new(0);
    if barriers.is_null() || n == 0 {
        return;
    }
    let mut out: Vec<[u32; 8]> = Vec::with_capacity(n as usize);
    for k in 0..n as usize {
        let mut b = barriers.add(k).read_unaligned();
        let (ty, flags) = (b[0], b[1]);
        if ty > 2 || flags > 3 {
            if N.fetch_add(1, Ordering::Relaxed) < 16 {
                crate::xlog_warn!("D3D12.X: ResourceBarrier: console barrier type {ty} flags {flags:#x} dropped ({:x?})", b);
            }
            continue;
        }
        if ty == 1 {
            // aliasing (resource before +8, after +16): on the PC only between placed / reserved
            // resources, and the console's texture memory is mostly XCL's committed resources
            // (the list would fail to Close) – an aliasing barrier for any resources instead
            b[2..6].fill(0);
        }
        if ty == 0 {
            // transition: resource (+8), subresource (+16), before (+20), after (+24)
            let (before, after) = (b[5], b[6]);
            if (before | after) & !PC_STATES != 0 && N.fetch_add(1, Ordering::Relaxed) < 16 {
                crate::xlog_warn!("D3D12.X: ResourceBarrier: console state bits {:#x} → {:#x} dropped", before & !PC_STATES, after & !PC_STATES);
            }
            b[5] &= PC_STATES;
            b[6] &= PC_STATES;
            if b[5] == b[6] {
                continue;
            }
        }
        out.push(b);
    }
    if !out.is_empty() {
        pc::<unsafe extern "system" fn(*mut c_void, u32, *const [u32; 8])>(this, PC_RESOURCE_BARRIER)(this, out.len() as u32, out.as_ptr());
    }
}

/// DMA list #9 Close: as on the PC; a failure is logged (the title stops on it)
pub(super) unsafe extern "system" fn dma_close_x(this: *mut c_void) -> i32 {
    let hr = pc::<unsafe extern "system" fn(*mut c_void) -> i32>(this, 9)(this);
    if hr < 0 {
        let ty = pc::<unsafe extern "system" fn(*mut c_void) -> i32>(this, PC_GET_TYPE)(this);
        crate::xlog_warn!("D3D12.X: DMA list Close → {:#x} (PC list type {ty})", hr as u32);
    }
    hr
}

/// a copy location with the offset of a buffer placed below its console address added
unsafe fn pc_location(l: *const D3D12_TEXTURE_COPY_LOCATION) -> Option<D3D12_TEXTURE_COPY_LOCATION> {
    let mut l = std::ptr::read(l.as_ref()?);
    if l.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT {
        l.Anonymous.PlacedFootprint.Offset += real::buffer_delta(transmute_copy::<_, usize>(&l.pResource));
    }
    Some(l)
}

/// CopyBufferRegion(dst, dst offset, src, src offset, bytes) (PC #15; DMA list #11)
pub(super) unsafe extern "system" fn copy_buffer_region_x(this: *mut c_void, dst: usize, dst_off: u64, src: usize, src_off: u64, n: u64) {
    pc::<unsafe extern "system" fn(*mut c_void, usize, u64, usize, u64, u64)>(this, PC_COPY_BUFFER_REGION)(this, dst, dst_off + real::buffer_delta(dst), src, src_off + real::buffer_delta(src), n)
}

/// CopyTextureRegion(dst, x, y, z, src, box) (PC #16; DMA list #12)
pub(super) unsafe extern "system" fn copy_texture_region_x(this: *mut c_void, dst: *const D3D12_TEXTURE_COPY_LOCATION, x: u32, y: u32, z: u32, src: *const D3D12_TEXTURE_COPY_LOCATION, bx: *const D3D12_BOX) {
    if let Some(d) = dst.as_ref() {
        super::present::note_use(transmute_copy::<_, usize>(&d.pResource), super::present::USE_COPY);
    }
    let (Some(d), Some(s)) = (pc_location(dst), pc_location(src)) else { return };
    pc::<unsafe extern "system" fn(*mut c_void, *const D3D12_TEXTURE_COPY_LOCATION, u32, u32, u32, *const D3D12_TEXTURE_COPY_LOCATION, *const D3D12_BOX)>(this, PC_COPY_TEXTURE_REGION)(this, &d, x, y, z, &s, bx);
    std::mem::forget((d, s)); // borrowed resource pointers
}

/// command list #83 CopyTextureRegionX(dst, x, y, z, src, box, flags): CopyTextureRegion with
/// console copy flags (the title's texture streaming calls it where the DMA list gets #12)
pub(super) unsafe extern "system" fn copy_texture_region_flags_x(this: *mut c_void, dst: *const D3D12_TEXTURE_COPY_LOCATION, x: u32, y: u32, z: u32, src: *const D3D12_TEXTURE_COPY_LOCATION, bx: *const D3D12_BOX, _flags: u32) {
    copy_texture_region_x(this, dst, x, y, z, src, bx)
}

/// CopyResource(dst, src) (PC #17; DMA list #13): buffers placed below their console address
/// are copied as their own bytes (the PC resources also hold what lies before them)
pub(super) unsafe extern "system" fn copy_resource_x(this: *mut c_void, dst: usize, src: usize) {
    super::present::note_use(dst, super::present::USE_COPY);
    let (dd, sd) = (real::buffer_delta(dst), real::buffer_delta(src));
    if dd == 0 && sd == 0 {
        return pc::<unsafe extern "system" fn(*mut c_void, usize, usize)>(this, PC_COPY_RESOURCE)(this, dst, src);
    }
    let n = real::console_buffer(dst).or(real::console_buffer(src)).map_or(0, |b| b.1 as u64);
    copy_buffer_region_x(this, dst, 0, src, 0, n);
}

/// command list #28 BeginQuery(heap, type, index) (PC #52)
pub(super) unsafe extern "system" fn begin_query_x(this: *mut c_void, heap: usize, ty: D3D12_QUERY_TYPE, index: u32) {
    static N: AtomicU64 = AtomicU64::new(0);
    if is_occlusion(ty) {
        let outer = LISTS
            .lock()
            .unwrap()
            .get_or_insert_with(HashMap::new)
            .entry(this as usize)
            .or_default()
            .occlusion
            .begin((heap, index), ty);
        if let Some(((h, i), oty)) = outer {
            if once(&N, 1) {
                crate::xlog_info!("D3D12.X: nested occlusion query – the outer one ends there and reports visible, the inner one counts");
            }
            pc::<QueryFn>(this, PC_END_QUERY)(this, h, oty, i);
        }
        if let Some(m) = VISIBLE.lock().unwrap().as_mut() {
            m.remove(&(heap, index));
        }
    }
    pc::<QueryFn>(this, PC_BEGIN_QUERY)(this, heap, ty, index);
}

/// command list #29 EndQuery(heap, type, index) (PC #53)
pub(super) unsafe extern "system" fn end_query_x(this: *mut c_void, heap: usize, ty: D3D12_QUERY_TYPE, index: u32) {
    if is_occlusion(ty) {
        let forward = LISTS
            .lock()
            .unwrap()
            .get_or_insert_with(HashMap::new)
            .entry(this as usize)
            .or_default()
            .occlusion
            .end((heap, index));
        if !forward {
            let v = if ty == D3D12_QUERY_TYPE_BINARY_OCCLUSION { 1 } else { VISIBLE_SAMPLES };
            VISIBLE.lock().unwrap().get_or_insert_with(HashMap::new).insert((heap, index), v);
            return;
        }
    }
    pc::<QueryFn>(this, PC_END_QUERY)(this, heap, ty, index);
}

/// command list #30 ResolveQueryData(heap, type, first, count, destination, offset) (PC #54):
/// outer occlusion queries ended early are written over the resolved data (after it, in COPY_DEST as
/// ResolveQueryData wants the destination)
pub(super) unsafe extern "system" fn resolve_query_data_x(this: *mut c_void, heap: usize, ty: D3D12_QUERY_TYPE, first: u32, count: u32, dest: usize, offset: u64) {
    type F = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32, u32, usize, u64);
    pc::<F>(this, PC_RESOLVE_QUERY_DATA)(this, heap, ty, first, count, dest, offset);
    if !is_occlusion(ty) {
        return;
    }
    let visible: Vec<(u32, u64)> = match VISIBLE.lock().unwrap().as_ref() {
        Some(m) if !m.is_empty() => (first..first.saturating_add(count)).filter_map(|i| m.get(&(heap, i)).map(|&v| (i, v))).collect(),
        _ => return,
    };
    let dest = dest as *mut c_void;
    let Some(res) = ID3D12Resource::from_raw_borrowed(&dest).filter(|_| !visible.is_empty()) else { return };
    let at = res.GetGPUVirtualAddress() + offset;
    let params: Vec<D3D12_WRITEBUFFERIMMEDIATE_PARAMETER> = visible
        .iter()
        .flat_map(|&(i, v)| {
            let a = at + 8 * (i - first) as u64;
            [D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: a, Value: v as u32 }, D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: a + 4, Value: (v >> 32) as u32 }]
        })
        .collect();
    let modes = vec![D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT; params.len()];
    type W = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER, *const D3D12_WRITEBUFFERIMMEDIATE_MODE);
    pc::<W>(this, PC_WRITE_BUFFER_IMMEDIATE)(this, params.len() as u32, params.as_ptr(), modes.as_ptr());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn nested_occlusion_queries_end_the_outer_one_early() {
        let ty = D3D12_QUERY_TYPE_OCCLUSION;
        let mut o = Occlusion::default();
        assert_eq!(o.begin((1, 0), ty), None);
        assert_eq!(o.begin((1, 1), ty), Some(((1, 0), ty)), "the outer one ends on the PC");
        assert!(o.end((1, 1)), "inner: a real query");
        // more per-object queries inside the same outer one
        assert_eq!(o.begin((1, 2), ty), None);
        assert!(o.end((1, 2)));
        assert!(!o.end((1, 0)), "outer: already ended");
        // overlapping, not nested: A, B, end A, end B
        assert_eq!(o.begin((1, 3), ty), None);
        assert_eq!(o.begin((1, 4), ty), Some(((1, 3), ty)));
        assert!(!o.end((1, 3)) && o.end((1, 4)));
        assert!(o.active.is_none() && o.ended_early.is_empty());
        assert!(o.end((1, 9)), "a query not begun here: the PC's call");
        // kickoff ends active query early so EndQuery later resolves it as visible
        assert_eq!(o.begin((1, 5), D3D12_QUERY_TYPE_BINARY_OCCLUSION), None);
        assert_eq!(o.end_active_early(), Some(((1, 5), D3D12_QUERY_TYPE_BINARY_OCCLUSION)));
        assert!(!o.end((1, 5)));
    }

    #[test]
    fn fill_patterns_repeat_the_value() {
        assert_eq!(pattern(&0x1122_3344u32.to_le_bytes())[..8], [0x44, 0x33, 0x22, 0x11, 0x44, 0x33, 0x22, 0x11]);
        assert_eq!(pattern(&[7u8; 16]), [7u8; 16]);
    }

    #[test]
    fn wait_comparison_ops_cover_all_d3d12_funcs() {
        assert_eq!(compare_u64(0, 1, 1), None);
        assert_eq!(compare_u64(9, 1, 1), None);
        assert_eq!(compare_u64(1, 5, 5), Some(false)); // NEVER
        assert_eq!(compare_u64(2, 4, 5), Some(true));  // LESS
        assert_eq!(compare_u64(2, 5, 5), Some(false));
        assert_eq!(compare_u64(3, 5, 5), Some(true));  // EQUAL
        assert_eq!(compare_u64(3, 4, 5), Some(false));
        assert_eq!(compare_u64(4, 5, 5), Some(true));  // LESS_EQUAL
        assert_eq!(compare_u64(4, 6, 5), Some(false));
        assert_eq!(compare_u64(5, 6, 5), Some(true));  // GREATER
        assert_eq!(compare_u64(5, 5, 5), Some(false));
        assert_eq!(compare_u64(6, 4, 5), Some(true));  // NOT_EQUAL
        assert_eq!(compare_u64(6, 5, 5), Some(false));
        assert_eq!(compare_u64(7, 5, 5), Some(true));  // GREATER_EQUAL
        assert_eq!(compare_u64(7, 4, 5), Some(false));
        assert_eq!(compare_u64(8, 0, 1), Some(true));  // ALWAYS
    }

    #[test]
    fn pending_wait_32_and_64_bit_with_mask_and_cross_thread_signal() {
        let cell32 = Arc::new(AtomicU32::new(0xDEAD_0000));
        let w32 = PendingWait {
            va: Arc::as_ptr(&cell32) as u64,
            width: 4,
            comparison: D3D12_COMPARISON_FUNC_EQUAL.0 as u32,
            value: 0x0000_005A,
            mask: 0x0000_00FF,
        };
        unsafe {
            assert!(!w32.is_satisfied());
            let writer = cell32.clone();
            let t = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(5));
                writer.store(0xBEEF_005A, Ordering::Release);
            });
            assert!(w32.wait_until_satisfied(Duration::from_secs(1)));
            t.join().unwrap();
        }

        let cell64 = Arc::new(AtomicU64::new(0x0000_0001_0000_0000));
        let w64 = PendingWait {
            va: Arc::as_ptr(&cell64) as u64,
            width: 8,
            comparison: D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32,
            value: 0x0000_0002_0000_0000,
            mask: u64::MAX,
        };
        unsafe {
            assert!(!w64.is_satisfied());
            // bounded timeout returns false without hanging
            assert!(!w64.wait_until_satisfied(Duration::from_millis(5)));
            cell64.store(0x0000_0002_0000_0001, Ordering::Release);
            assert!(w64.is_satisfied());
        }
    }

    #[test]
    fn written_ranges_detect_same_list_overlap() {
        let mut s = ListState::default();
        s.note_write(0x1000, 8);
        assert!(s.has_prior_write(0x1000, 4));
        assert!(s.has_prior_write(0x1004, 4));
        assert!(s.has_prior_write(0x0FFC, 8));
        assert!(!s.has_prior_write(0x1008, 4));
        assert!(!s.has_prior_write(0x0FF8, 8));
    }

    #[test]
    fn execute_with_waits_splits_batches_before_unsatisfied_list() {
        // Use synthetic non-zero list pointers distinct from real COM objects.
        let l0 = 0x7000_0001usize as *mut c_void;
        let l1 = 0x7000_0002usize as *mut c_void;
        let l2 = 0x7000_0003usize as *mut c_void;
        let cell = Box::new(AtomicU32::new(0));
        let va = &*cell as *const AtomicU32 as u64;

        unsafe {
            on_create(l0, 0x100, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
            on_create(l1, 0x100, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
            on_create(l2, 0x100, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
        }
        {
            let mut g = LISTS.lock().unwrap();
            let s = g.as_mut().unwrap().get_mut(&(l1 as usize)).unwrap();
            s.push_wait(PendingWait {
                va,
                width: 4,
                comparison: D3D12_COMPARISON_FUNC_EQUAL.0 as u32,
                value: 42,
                mask: u32::MAX as u64,
            });
        }

        let mut batches: Vec<Vec<usize>> = Vec::new();
        unsafe {
            execute_with_waits(&[l0, l1, l2], |chunk| {
                batches.push(chunk.iter().map(|&p| p as usize).collect());
                // Simulate l0's GPU execution satisfying l1's wait!
                if chunk.contains(&l0) {
                    cell.store(42, Ordering::Release);
                }
            });
        }
        assert_eq!(batches, vec![vec![l0 as usize], vec![l1 as usize, l2 as usize]]);

        // Now that cell == 42 is already satisfied, the whole batch submits in one call.
        batches.clear();
        unsafe {
            execute_with_waits(&[l0, l1, l2], |chunk| {
                batches.push(chunk.iter().map(|&p| p as usize).collect());
            });
        }
        assert_eq!(batches, vec![vec![l0 as usize, l1 as usize, l2 as usize]]);

        on_release(l0 as usize);
        on_release(l1 as usize);
        on_release(l2 as usize);
    }

    #[test]
    fn queue_picking_prefers_last_queue_and_respects_release() {
        let dev = 0x8000_0001usize;
        let q1 = 0x8000_0010usize;
        let q2 = 0x8000_0020usize;
        let q_copy = 0x8000_0030usize;
        on_queue_create(q1, dev, D3D12_COMMAND_LIST_TYPE_DIRECT);
        on_queue_create(q2, dev, D3D12_COMMAND_LIST_TYPE_DIRECT);
        on_queue_create(q_copy, dev, D3D12_COMMAND_LIST_TYPE_COPY);

        // Most recently created matching queue wins when no last_queue
        assert_eq!(pick_queue(None, dev, D3D12_COMMAND_LIST_TYPE_DIRECT), Some(q2));
        assert_eq!(pick_queue(None, dev, D3D12_COMMAND_LIST_TYPE_COPY), Some(q_copy));
        // Explicit last_queue wins when still live
        assert_eq!(pick_queue(Some(q1), dev, D3D12_COMMAND_LIST_TYPE_DIRECT), Some(q1));
        // Releasing q1 falls back to q2
        on_queue_release(q1);
        assert_eq!(pick_queue(Some(q1), dev, D3D12_COMMAND_LIST_TYPE_DIRECT), Some(q2));
        on_queue_release(q2);
        on_queue_release(q_copy);
        assert_eq!(pick_queue(None, dev, D3D12_COMMAND_LIST_TYPE_DIRECT), None);
    }

    #[test]
    fn depth_bias_state_tracks_fallback_and_clear() {
        let l = 0x9000_0001usize as *mut c_void;
        let key = l as usize;
        unsafe {
            on_create(l, 0x100, 0xAAA, D3D12_COMMAND_LIST_TYPE_DIRECT);
            on_set_pipeline_state(l, 0xBBB);
        }
        {
            let mut g = LISTS.lock().unwrap();
            let s = g.as_mut().unwrap().get_mut(&key).unwrap();
            assert_eq!(s.bound_pso, 0xBBB);
            assert!(!s.pso_dirty);
            s.depth_bias = Some(DepthBias::new(-10.0, 0.0, 1.0));
            s.pso_dirty = true;
        }
        // Switching to a non-dynamic PSO while depth bias is active keeps pso_dirty for fallback
        unsafe { on_set_pipeline_state(l, 0xCCC) };
        {
            let g = LISTS.lock().unwrap();
            let s = g.as_ref().unwrap().get(&key).unwrap();
            assert_eq!(s.bound_pso, 0xCCC);
            assert!(s.pso_dirty);
        }
        // ClearState resets depth bias, native_bias, and dirty flag
        unsafe { on_clear_state(l, 0xDDD) };
        {
            let g = LISTS.lock().unwrap();
            let s = g.as_ref().unwrap().get(&key).unwrap();
            assert_eq!(s.bound_pso, 0xDDD);
            assert!(s.depth_bias.is_none());
            assert_eq!(s.native_bias, DepthBias::ZERO);
            assert!(!s.pso_dirty);
        }
        on_release(key);
    }

    #[test]
    fn depth_bias_native_list9_and_bundle_fallback() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let supported = device_x::probe_dynamic_depth_bias(&dev);
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
            let has_list9 = list.cast::<ID3D12GraphicsCommandList9>().is_ok();
            let raw = list.as_raw();
            let key = raw as usize;

            let pso_dyn_zero = 0x9100_0001usize;
            let pso_dyn_static = 0x9100_0002usize;
            let pso_static_only = 0x9100_0003usize;
            let static_bias = DepthBias::new(-500.0, 0.0, 0.0);
            device_x::register_test_base(pso_dyn_zero, true, DepthBias::ZERO);
            device_x::register_test_base(pso_dyn_static, true, static_bias);
            device_x::register_test_base(pso_static_only, false, DepthBias::ZERO);

            let cache_before = device_x::depth_bias_cache_len();
            on_create(raw, alloc.as_raw() as usize, pso_dyn_zero, D3D12_COMMAND_LIST_TYPE_DIRECT);

            if supported && has_list9 {
                // Fractional f32 depth bias takes the native ID3D12GraphicsCommandList9 path
                // without rounding to i32 or dirtying the patched-PSO fallback cache.
                set_depth_bias_x(raw, -125.75, 0.05, 1.25);
                {
                    let g = LISTS.lock().unwrap();
                    let s = g.as_ref().unwrap().get(&key).unwrap();
                    assert_eq!(s.depth_bias, Some(DepthBias::new(-125.75, 0.05, 1.25)));
                    assert_eq!(s.native_bias, DepthBias::new(-125.75, 0.05, 1.25));
                    assert_eq!(s.native_bias.depth_bias(), -125.75);
                    assert!(!s.pso_dirty);
                }
                assert_eq!(device_x::depth_bias_cache_len(), cache_before);

                // Resetting depth bias to 0 updates native_bias to ZERO without touching fallback cache.
                set_depth_bias_x(raw, 0.0, 0.0, 0.0);
                {
                    let g = LISTS.lock().unwrap();
                    let s = g.as_ref().unwrap().get(&key).unwrap();
                    assert!(s.depth_bias.is_none());
                    assert_eq!(s.native_bias, DepthBias::ZERO);
                    assert!(!s.pso_dirty);
                }

                // Binding a dynamic PSO with non-zero static bias syncs native_bias to its static bias.
                on_set_pipeline_state(raw, pso_dyn_static);
                {
                    let g = LISTS.lock().unwrap();
                    let s = g.as_ref().unwrap().get(&key).unwrap();
                    assert_eq!(s.native_bias, static_bias);
                    assert!(!s.pso_dirty);
                }

                // Switching back to a zero-static-bias dynamic PSO restores native_bias to ZERO.
                on_set_pipeline_state(raw, pso_dyn_zero);
                {
                    let g = LISTS.lock().unwrap();
                    let s = g.as_ref().unwrap().get(&key).unwrap();
                    assert_eq!(s.native_bias, DepthBias::ZERO);
                    assert!(!s.pso_dirty);
                }
            }

            // Non-dynamic PSO must fall back to the patched-PSO cache (pso_dirty = true).
            on_set_pipeline_state(raw, pso_static_only);
            set_depth_bias_x(raw, -50.0, 0.0, 0.0);
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&key).unwrap();
                assert_eq!(s.depth_bias, Some(DepthBias::new(-50.0, 0.0, 0.0)));
                assert!(s.pso_dirty);
            }
            on_release(key);

            // Bundle lists cannot call ID3D12GraphicsCommandList9::RSSetDepthBias and must fall back.
            let b_raw = 0x9100_00F0usize as *mut c_void;
            on_create(b_raw, 0x100, pso_dyn_zero, D3D12_COMMAND_LIST_TYPE_BUNDLE);
            set_depth_bias_x(b_raw, -75.0, 0.0, 0.0);
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&(b_raw as usize)).unwrap();
                assert!(s.pso_dirty);
                assert_eq!(s.native_bias, DepthBias::ZERO);
            }
            on_release(b_raw as usize);

            device_x::forget_base(pso_dyn_zero);
            device_x::forget_base(pso_dyn_static);
            device_x::forget_base(pso_static_only);
        }
    }

    #[test]
    fn depth_bias_native_path_exact_f32_and_list9_fallback() {
        use std::cell::RefCell;
        use std::rc::Rc;

        // Verify PC vtable slot of ID3D12GraphicsCommandList9::RSSetDepthBias is in layout table.
        let cl9 = super::super::layout::find("ID3D12GraphicsCommandList9").expect("ID3D12GraphicsCommandList9 in layout");
        assert_eq!(super::super::layout::slot(cl9, "RSSetDepthBias"), Some(82));

        let calls: Rc<RefCell<Vec<(usize, f32, f32, f32)>>> = Rc::new(RefCell::new(Vec::new()));
        let allow = Rc::new(std::cell::Cell::new(true));
        let calls_cb = calls.clone();
        let allow_cb = allow.clone();
        TEST_NATIVE_BIAS.with(|h| {
            *h.borrow_mut() = Some(Box::new(move |list, bias| {
                if !allow_cb.get() {
                    return false;
                }
                calls_cb.borrow_mut().push((list, bias.depth_bias(), bias.depth_bias_clamp(), bias.slope_scaled_depth_bias()));
                true
            }));
        });

        let l = 0x9200_0001usize as *mut c_void;
        let key = l as usize;
        let pso_dyn_zero = 0x9200_0010usize;
        let pso_dyn_static = 0x9200_0011usize;
        let pso_static_only = 0x9200_0012usize;
        device_x::register_test_base(pso_dyn_zero, true, DepthBias::ZERO);
        device_x::register_test_base(pso_dyn_static, true, DepthBias::new(-500.0, 0.25, 2.0));
        device_x::register_test_base(pso_static_only, false, DepthBias::ZERO);

        let cache_before = device_x::depth_bias_cache_len();
        unsafe {
            // 1. Initial PSO with baked static bias seeds native RSSetDepthBias on CreateCommandList.
            on_create(l, 0x100, pso_dyn_static, D3D12_COMMAND_LIST_TYPE_DIRECT);
            assert_eq!(*calls.borrow(), vec![(key, -500.0, 0.25, 2.0)]);

            // 2. Switching to zero-static-bias dynamic PSO resets native RSSetDepthBias to (0, 0, 0).
            on_set_pipeline_state(l, pso_dyn_zero);
            assert_eq!(calls.borrow().last(), Some(&(key, 0.0, 0.0, 0.0)));

            // 3. Fractional f32 override is forwarded verbatim (no i32 rounding, no fallback PSO cache).
            set_depth_bias_x(l, -1000.5, 0.05, 1.5);
            assert_eq!(calls.borrow().last(), Some(&(key, -1000.5, 0.05, 1.5)));
            assert_eq!(device_x::depth_bias_cache_len(), cache_before);
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&key).unwrap();
                assert!(!s.pso_dirty);
                assert_eq!(s.native_bias, DepthBias::new(-1000.5, 0.05, 1.5));
            }

            // 4. Switching between dynamic PSOs while override is active keeps the override without redundant calls.
            let n_before = calls.borrow().len();
            on_set_pipeline_state(l, pso_dyn_static);
            assert_eq!(calls.borrow().len(), n_before);

            // 5. Resetting to (0, 0, 0) sets native bias to (0, 0, 0); ClearState with static-bias PSO re-seeds it.
            set_depth_bias_x(l, 0.0, 0.0, 0.0);
            assert_eq!(calls.borrow().last(), Some(&(key, 0.0, 0.0, 0.0)));
            on_clear_state(l, pso_dyn_static);
            assert_eq!(calls.borrow().last(), Some(&(key, -500.0, 0.25, 2.0)));

            // 6. If ID3D12GraphicsCommandList9 is unavailable, set_depth_bias_x falls back cleanly to pso_dirty.
            on_set_pipeline_state(l, pso_dyn_zero);
            allow.set(false);
            set_depth_bias_x(l, -42.5, 0.0, 0.0);
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&key).unwrap();
                assert!(s.pso_dirty);
                assert_eq!(s.depth_bias, Some(DepthBias::new(-42.5, 0.0, 0.0)));
            }

            // 7. Non-dynamic PSO always uses the patched-PSO fallback and never invokes RSSetDepthBias.
            allow.set(true);
            let n_before = calls.borrow().len();
            on_set_pipeline_state(l, pso_static_only);
            set_depth_bias_x(l, -10.0, 0.0, 0.0);
            assert_eq!(calls.borrow().len(), n_before);
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&key).unwrap();
                assert!(s.pso_dirty);
            }
        }

        TEST_NATIVE_BIAS.with(|h| *h.borrow_mut() = None);
        on_release(key);
        device_x::forget_base(pso_dyn_zero);
        device_x::forget_base(pso_dyn_static);
        device_x::forget_base(pso_static_only);
    }
}
