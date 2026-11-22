//! The title's own memory, seen by the GPU: sparse windows.
//!
//! A console title keeps GPU data in memory it reserves and commits itself (XDefiant: one
//! 8 GB reservation of 64 KiB pages it commits in steps and places buffers in) – on the
//! console all of it is GPU-visible at the same addresses. XCL gives such reservations
//! addresses in the *sparse windows* of the unified-memory block (`gpu::uma`: windows after
//! the committed one, reserved only), so translated shaders reach them through `mem1`…`mem3`
//! like any unified memory. On the GPU each sparse window is a reserved (tiled) buffer of
//! 4 GiB whose 64 KiB tiles are mapped onto video-memory heaps as the title commits pages.
//! The CPU's view stays the title's memory: what the CPU wrote (write-watch) is copied into
//! the tiles before the title's next ExecuteCommandLists, and its queues wait for that copy.
//! Only the bytes the CPU changed go (64-byte chunks against a hash of what was sent last,
//! `changed_chunks`): the rest of a written page may hold what the GPU wrote there (counters,
//! indirect arguments) and must not be overwritten with the CPU's stale view of it.
//! Buffers the title places there get reserved resources over the same tiles, so PC-side
//! copies, views and barriers see the same memory. GPU writes are not copied back to the
//! CPU's view (not needed by rendering so far).

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

const TILE: usize = 0x1_0000;
/// tiles of a video-memory pool heap (64 MiB)
const POOL_TILES: u32 = 1024;
/// bytes of the upload staging buffer
const STAGING: usize = 64 << 20;

struct State {
    dev: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    value: u64,
    /// (window start, reserved buffer) of each sparse window
    windows: Vec<(usize, ID3D12Resource)>,
    pools: Vec<ID3D12Heap>,
    free: Vec<(u32, u32)>,
    /// tile address → (pool, tile in it)
    mapped: HashMap<usize, (u32, u32)>,
    /// pool tiles also used by placed buffers (not freed while they are)
    pinned: HashMap<(u32, u32), u32>,
    /// placed buffers → the pool tiles they use
    buffers: HashMap<usize, Vec<(u32, u32)>>,
    /// reused tiles to clear: (address, bytes)
    to_zero: Vec<(usize, usize)>,
    staging: Option<(ID3D12Resource, *mut u8)>,
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    /// the fence value each title queue last waited for
    waited: HashMap<usize, u64>,
    /// 4 KiB page → hash of each 64-byte chunk the GPU got from the CPU (absent: all zero)
    sent: HashMap<usize, Box<[u64; CHUNKS]>>,
}

const PAGE: usize = 0x1000;
const CHUNK: usize = 64;
const CHUNKS: usize = PAGE / CHUNK;

/// hash of 64 bytes
fn chunk_hash(c: &[u8]) -> u64 {
    let mut h = 0x243F_6A88_85A3_08D3u64;
    for w in c.chunks_exact(8) {
        h = (h ^ u64::from_le_bytes(w.try_into().unwrap())).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29);
    }
    h
}

fn zero_hashes() -> [u64; CHUNKS] {
    [chunk_hash(&[0u8; CHUNK]); CHUNKS]
}

/// The byte ranges (offset, length) of `page` whose 64-byte chunks differ from `sent`
/// (updated to the page's chunks).
pub fn changed_chunks(page: &[u8], sent: &mut [u64; CHUNKS]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (k, c) in page.chunks_exact(CHUNK).enumerate() {
        let h = chunk_hash(c);
        if h == sent[k] {
            continue;
        }
        sent[k] = h;
        match out.last_mut() {
            Some(last) if last.0 + last.1 == k * CHUNK => last.1 += CHUNK,
            _ => out.push((k * CHUNK, CHUNK)),
        }
    }
    out
}

// D3D12 objects are free-threaded; the staging pointer is only used under the lock
unsafe impl Send for State {}

static STATE: Mutex<Option<State>> = Mutex::new(None);
/// commits before the device existed
static EARLY: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

fn buffer_desc(bytes: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    }
}

/// The sparse windows' buffers on `dev`: (buffer, GPU address) for each window after the
/// committed ones – the shader ABI's `mem1`… (empty: no sparse windows).
pub unsafe fn init(dev: &ID3D12Device) -> Vec<(usize, u64)> {
    let Some(u) = crate::gpu::uma::get() else { return Vec::new() };
    let mut out = Vec::new();
    let mut windows = Vec::new();
    for k in u.windows()..u.shader_windows() {
        let (base, size) = u.window(k);
        let mut r: Option<ID3D12Resource> = None;
        if let Err(e) = dev.CreateReservedResource(&buffer_desc(size as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut r) {
            crate::xlog_warn!("sparse window {k}: CreateReservedResource: {e}");
            return Vec::new();
        }
        let Some(r) = r else { return Vec::new() };
        out.push((r.clone().into_raw() as usize, r.GetGPUVirtualAddress()));
        windows.push((base, r));
    }
    let made = (|| -> windows::core::Result<State> {
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
        list.Close()?;
        Ok(State {
            dev: dev.clone(),
            queue,
            fence,
            value: 0,
            windows,
            pools: Vec::new(),
            free: Vec::new(),
            mapped: HashMap::new(),
            pinned: HashMap::new(),
            buffers: HashMap::new(),
            to_zero: Vec::new(),
            staging: None,
            alloc,
            list,
            waited: HashMap::new(),
            sent: HashMap::new(),
        })
    })();
    match made {
        Ok(s) => {
            crate::xlog_info!("D3D12.X: {} sparse window(s) for the title's own memory (tiled buffers on video memory)", out.len());
            *STATE.lock().unwrap() = Some(s);
            for (a, n) in std::mem::take(&mut *EARLY.lock().unwrap()) {
                committed(a, n);
            }
            out
        }
        Err(e) => {
            crate::xlog_warn!("sparse windows: {e}");
            Vec::new()
        }
    }
}

impl State {
    fn window(&self, a: usize) -> Option<(usize, &ID3D12Resource)> {
        self.windows.iter().find(|(b, _)| a >= *b && a < *b + crate::gpu::uma::WINDOW).map(|(b, r)| (*b, r))
    }

    unsafe fn take_tile(&mut self) -> Option<((u32, u32), bool)> {
        if let Some(t) = self.free.pop() {
            return Some((t, true));
        }
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: POOL_TILES as u64 * TILE as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES,
        };
        let mut heap: Option<ID3D12Heap> = None;
        if let Err(e) = self.dev.CreateHeap(&desc, &mut heap) {
            crate::xlog_warn!("sparse windows: no video memory for more tiles ({e})");
            return None;
        }
        let h = self.pools.len() as u32;
        self.pools.push(heap?);
        self.free.extend((1..POOL_TILES).rev().map(|t| (h, t)));
        Some(((h, 0), false))
    }

    /// map `tiles` (tile addresses) of `res` (starting at `res_base`) onto pool tiles
    unsafe fn map_tiles(&self, res: &ID3D12Resource, res_base: usize, tiles: &[(usize, (u32, u32))]) {
        for &(a, (h, t)) in tiles {
            let start = D3D12_TILED_RESOURCE_COORDINATE { X: ((a - res_base) / TILE) as u32, ..Default::default() };
            let region = D3D12_TILE_REGION_SIZE { NumTiles: 1, UseBox: false.into(), ..Default::default() };
            let (flags, n) = (D3D12_TILE_RANGE_FLAG_NONE, 1u32);
            self.queue.UpdateTileMappings(res, 1, Some(&start), Some(&region), &self.pools[h as usize], 1, Some(&flags), Some(&t), Some(&n), D3D12_TILE_MAPPING_FLAG_NONE);
        }
    }

    unsafe fn unmap_tiles(&self, res: &ID3D12Resource, res_base: usize, tiles: &[usize]) {
        for &a in tiles {
            let start = D3D12_TILED_RESOURCE_COORDINATE { X: ((a - res_base) / TILE) as u32, ..Default::default() };
            let region = D3D12_TILE_REGION_SIZE { NumTiles: 1, UseBox: false.into(), ..Default::default() };
            let flags = D3D12_TILE_RANGE_FLAG_NULL;
            self.queue.UpdateTileMappings(res, 1, Some(&start), Some(&region), None, 1, Some(&flags), None, None, D3D12_TILE_MAPPING_FLAG_NONE);
        }
    }

    /// tiles of `[a, a + n)` get video memory (mapped in their window)
    unsafe fn ensure_mapped(&mut self, a: usize, n: usize) {
        let first = a & !(TILE - 1);
        let mut new: Vec<(usize, (u32, u32))> = Vec::new();
        let mut t = first;
        while t < a + n {
            if !self.mapped.contains_key(&t) && self.window(t).is_some() {
                let Some((tile, reused)) = self.take_tile() else { break };
                self.mapped.insert(t, tile);
                if reused {
                    self.to_zero.push((t, TILE));
                }
                new.push((t, tile));
            }
            t += TILE;
        }
        // per window
        let mut by_window: HashMap<usize, Vec<(usize, (u32, u32))>> = HashMap::new();
        for x in new {
            if let Some((b, _)) = self.window(x.0) {
                by_window.entry(b).or_default().push(x);
            }
        }
        for (b, tiles) in by_window {
            if let Some((_, r)) = self.window(b) {
                let r = r.clone();
                self.map_tiles(&r, b, &tiles);
            }
        }
        self.signal();
    }

    unsafe fn signal(&mut self) {
        self.value += 1;
        let _ = self.queue.Signal(&self.fence, self.value);
    }

    unsafe fn wait_idle(&self) {
        if self.fence.GetCompletedValue() < self.value {
            if let Ok(ev) = windows::Win32::System::Threading::CreateEventW(None, false, false, None) {
                let _ = self.fence.SetEventOnCompletion(self.value, ev);
                windows::Win32::System::Threading::WaitForSingleObject(ev, 5000);
                let _ = windows::Win32::Foundation::CloseHandle(ev);
            }
        }
    }

    unsafe fn staging(&mut self) -> Option<(ID3D12Resource, *mut u8)> {
        if self.staging.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(STAGING as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            self.dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut r).ok()?;
            let r = r?;
            let mut p = std::ptr::null_mut();
            r.Map(0, None, Some(&mut p)).ok()?;
            self.staging = Some((r, p as *mut u8));
        }
        self.staging.clone()
    }

    /// copy `runs` (address, bytes) of the CPU's view into the tiles (`zero`: clear instead)
    unsafe fn upload(&mut self, runs: &[(usize, usize, bool)]) {
        let Some((staging, p)) = self.staging() else { return };
        let mut used = 0usize;
        let mut open = false;
        for &(a, n, zero) in runs {
            let mut done = 0;
            while done < n {
                if used == STAGING {
                    self.flush(&mut open);
                    used = 0;
                }
                let chunk = (n - done).min(STAGING - used);
                let Some((b, r)) = self.window(a + done).map(|(b, r)| (b, r.clone())) else { break };
                if !open {
                    self.wait_idle();
                    let _ = self.alloc.Reset();
                    let _ = self.list.Reset(&self.alloc, None);
                    open = true;
                }
                if zero {
                    std::ptr::write_bytes(p.add(used), 0, chunk);
                } else {
                    std::ptr::copy_nonoverlapping((a + done) as *const u8, p.add(used), chunk);
                }
                self.list.CopyBufferRegion(&r, (a + done - b) as u64, &staging, used as u64, chunk as u64);
                used += chunk;
                done += chunk;
            }
        }
        if open {
            self.flush(&mut open);
        }
    }

    unsafe fn flush(&mut self, open: &mut bool) {
        if !*open {
            return;
        }
        *open = false;
        if self.list.Close().is_ok() {
            if let Ok(l) = self.list.cast::<ID3D12CommandList>() {
                self.queue.ExecuteCommandLists(&[Some(l)]);
            }
        }
        self.signal();
        // the staging buffer is written again only after this
        self.wait_idle();
    }
}

/// The title committed `[a, a + n)` in a sparse window.
pub unsafe fn committed(a: usize, n: usize) {
    let mut g = STATE.lock().unwrap();
    match g.as_mut() {
        Some(s) => s.ensure_mapped(a, n),
        None => EARLY.lock().unwrap().push((a, n)),
    }
}

/// The title decommitted `[a, a + n)`: its whole tiles give their video memory back (unless a
/// placed buffer still uses it).
pub unsafe fn decommitted(a: usize, n: usize) {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else {
        EARLY.lock().unwrap().retain(|&(x, m)| x + m <= a || x >= a + n);
        return;
    };
    let first = (a + TILE - 1) & !(TILE - 1);
    let tiles: Vec<usize> = (first..(a + n) & !(TILE - 1)).step_by(TILE).filter(|t| s.mapped.contains_key(t)).collect();
    let mut by_window: HashMap<usize, Vec<usize>> = HashMap::new();
    for &t in &tiles {
        if let Some((b, _)) = s.window(t) {
            by_window.entry(b).or_default().push(t);
        }
    }
    for (b, ts) in by_window {
        if let Some((_, r)) = s.window(b) {
            let r = r.clone();
            s.unmap_tiles(&r, b, &ts);
        }
    }
    for pg in (a & !(PAGE - 1)..a + n).step_by(PAGE) {
        s.sent.remove(&pg);
    }
    for t in tiles {
        if let Some(tile) = s.mapped.remove(&t) {
            if !s.pinned.contains_key(&tile) {
                s.free.push(tile);
            }
        }
    }
    s.signal();
}

impl State {
    /// a pool tile no window maps and no buffer uses goes back to the free list
    fn release_if_unused(&mut self, tile: (u32, u32)) {
        if !self.pinned.contains_key(&tile) && !self.mapped.values().any(|&x| x == tile) {
            self.free.push(tile);
        }
    }
}

/// Before the title's ExecuteCommandLists on `queue`: the pages the CPU wrote (`dirty`, 4 KiB
/// pages from the write-watch, sorted) and the reused tiles are copied into video memory, and
/// `queue` waits for that and for new tile mappings.
pub unsafe fn before_execute(queue: &ID3D12CommandQueue, dirty: &[usize], page: usize) {
    let t0 = std::time::Instant::now();
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    let zeroed = std::mem::take(&mut s.to_zero);
    for &(a, n) in &zeroed {
        for pg in (a..a + n).step_by(PAGE) {
            s.sent.remove(&pg);
        }
    }
    let mut runs: Vec<(usize, usize, bool)> = zeroed.into_iter().map(|(a, n)| (a, n, true)).collect();
    let mut i = 0;
    while i < dirty.len() {
        let a = dirty[i];
        let mut n = page;
        i += 1;
        while i < dirty.len() && dirty[i] == a + n {
            n += page;
            i += 1;
        }
        // only what has video memory (committed tiles of a sparse window)
        if s.window(a).is_some() {
            let mut x = a;
            while x < a + n {
                let t = x & !(TILE - 1);
                let end = (t + TILE).min(a + n);
                if s.mapped.contains_key(&t) {
                    // per page: only the chunks the CPU changed since they were sent
                    let mut pg = x & !(PAGE - 1);
                    while pg < end {
                        let bytes = std::slice::from_raw_parts(pg as *const u8, PAGE);
                        let sent = s.sent.entry(pg).or_insert_with(|| Box::new(zero_hashes()));
                        for (o, n) in changed_chunks(bytes, sent) {
                            match runs.last_mut() {
                                Some(last) if !last.2 && last.0 + last.1 == pg + o => last.1 += n,
                                _ => runs.push((pg + o, n, false)),
                            }
                        }
                        pg += PAGE;
                    }
                }
                x = end;
            }
        }
    }
    if !runs.is_empty() {
        s.upload(&runs);
    }
    {
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
        static CALLS: AtomicU64 = AtomicU64::new(0);
        static BYTES: AtomicU64 = AtomicU64::new(0);
        static MICROS: AtomicU64 = AtomicU64::new(0);
        let bytes: usize = runs.iter().map(|r| r.1).sum();
        let n = CALLS.fetch_add(1, Relaxed) + 1;
        let b = BYTES.fetch_add(bytes as u64, Relaxed) + bytes as u64;
        let us = MICROS.fetch_add(t0.elapsed().as_micros() as u64, Relaxed) + t0.elapsed().as_micros() as u64;
        if n.is_power_of_two() && n >= 16 {
            crate::xlog_info!("sparse windows: {n} submits, {} MB of CPU writes uploaded, {} ms spent, {} tiles mapped", b >> 20, us / 1000, s.mapped.len());
        }
    }
    let q = queue.as_raw() as usize;
    if s.value > 0 && s.waited.get(&q).copied().unwrap_or(0) < s.value {
        let _ = queue.Wait(&s.fence, s.value);
        s.waited.insert(q, s.value);
    }
}

/// A buffer the title places at `va` in a sparse window (`bytes` long, `desc` the PC
/// description): a reserved resource over the window's tiles there, starting at the 64 KiB
/// below `va` (returned: its distance from there).
pub unsafe fn place_buffer(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<(crate::util::HRESULT, usize)> {
    let mut g = STATE.lock().unwrap();
    let s = g.as_mut()?;
    let start = va & !(TILE - 1);
    let delta = va - start;
    let bytes = desc.Width as usize;
    s.window(start)?;
    s.window(va + bytes - 1)?;
    let mut d = *desc;
    d.Width = (delta + bytes) as u64;
    d.Alignment = 0;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    let hr = super::real::dev_create_reserved_resource(this, &d as *const _ as usize, state as usize, 0, riid, ppv);
    if hr < 0 || ppv.is_null() || (*ppv).is_null() {
        return None;
    }
    // its tiles: the window's video memory at the same addresses
    s.ensure_mapped(start, delta + bytes);
    let tiles: Vec<(usize, (u32, u32))> = (start..va + bytes).step_by(TILE).filter_map(|t| s.mapped.get(&t).map(|&x| (t, x))).collect();
    let raw = *ppv;
    let res = ID3D12Resource::from_raw_borrowed(&raw)?.clone();
    s.map_tiles(&res, start, &tiles);
    for &(_, tile) in &tiles {
        *s.pinned.entry(tile).or_default() += 1;
    }
    s.buffers.insert(raw as usize, tiles.iter().map(|x| x.1).collect());
    s.signal();
    Some((hr, delta))
}

/// A placed buffer is gone: its tiles are not pinned by it any more.
pub fn forget_buffer(res: usize) {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    let Some(tiles) = s.buffers.remove(&res) else { return };
    for tile in tiles {
        if let Some(c) = s.pinned.get_mut(&tile) {
            *c -= 1;
            if *c == 0 {
                s.pinned.remove(&tile);
                s.release_if_unused(tile);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the 64-byte chunks the CPU changed are sent again – the rest of the page may hold
    /// what the GPU wrote.
    #[test]
    fn only_changed_chunks_are_sent() {
        let mut page = vec![0u8; PAGE];
        let mut sent = zero_hashes();
        assert!(changed_chunks(&page, &mut sent).is_empty(), "a zero page is what the GPU has");
        page[10] = 1;
        page[200] = 2;
        page[255] = 3;
        page[4000] = 4;
        assert_eq!(changed_chunks(&page, &mut sent), vec![(0, 64), (192, 64), (3968, 64)]);
        assert!(changed_chunks(&page, &mut sent).is_empty(), "nothing changed since");
        page[256] = 5; // next to the chunk at 192: one run
        page[200] = 9;
        assert_eq!(changed_chunks(&page, &mut sent), vec![(192, 128)]);
    }
}
