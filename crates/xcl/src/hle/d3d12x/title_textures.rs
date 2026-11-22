//! Textures a title places in its own memory (CreatePlacedResourceX): on the console the GPU
//! samples that memory, laid out as the console's layout library says (`hle::xg`, checked
//! against the console's own). Here a PC texture holds a detiled copy: uploaded when the
//! texture is created and again before the title's next ExecuteCommandLists once the CPU wrote
//! any of its pages (unified memory and the title's graphics reservations are write-watched,
//! `gpu::uma`, `kernelx::VirtualAlloc`) – textures streamed in after creation (mips loaded
//! later, physical pages mapped in, DirectStorage) included.
//!
//! The copy waits for the GPU on an XCL queue of the device. Memory that is not readable
//! (reserved, not mapped yet) reads as zero. GPU writes into the title's memory (copies to the
//! address) are not seen; the title's own GPU work on the texture goes to the PC texture.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::gpu::texture::{self, Subresource};
use crate::hle::xg;

/// GetWriteWatch / ResetWriteWatch
const WRITE_WATCH_FLAG_RESET: u32 = 1;
#[link(name = "kernel32")]
extern "system" {
    fn GetWriteWatch(flags: u32, base: *const c_void, size: usize, addresses: *mut *mut c_void, count: *mut usize, granularity: *mut u32) -> u32;
    fn ResetWriteWatch(base: *const c_void, size: usize) -> u32;
}

/// (device, address, layout, state) of an upload
type Upload = (usize, usize, Arc<xg::Texture>, D3D12_RESOURCE_STATES);

struct Entry {
    device: usize,
    va: usize,
    tex: Arc<xg::Texture>,
    /// the state the texture was created in (uploads leave it there)
    state: D3D12_RESOURCE_STATES,
}

// raw COM pointers of the title's objects; uploads hold a reference while they run
unsafe impl Send for Entry {}

/// PC texture → its place in the title's memory
static TEXTURES: Mutex<BTreeMap<usize, Entry>> = Mutex::new(BTreeMap::new());
/// the queue uploads go to, per device
static QUEUES: Mutex<BTreeMap<usize, ID3D12CommandQueue>> = Mutex::new(BTreeMap::new());
static UPLOADS: AtomicU64 = AtomicU64::new(0);

/// Number of textures kept in title memory and uploads done so far.
pub fn counts() -> (usize, u64) {
    (TEXTURES.lock().unwrap().len(), UPLOADS.load(Ordering::Relaxed))
}

/// A texture the title placed at `va` (its console desc `desc`): kept in step with the memory.
pub(super) unsafe fn register(device: *mut c_void, res: *mut c_void, va: usize, desc: &[u8; 0x38], state: D3D12_RESOURCE_STATES) -> Result<(), String> {
    let tex = xg::Texture::new(desc).map_err(|hr| format!("layout {hr:#x}"))?;
    // render targets and depth buffers are the GPU's (their metadata planes, empty planes)
    if tex.planes.len() != 1 || tex.texels().is_none() {
        return Err("a render target / depth buffer".into());
    }
    let bytes = tex.size() as usize;
    let e = Entry { device: device as usize, va, tex: Arc::new(tex), state };
    if crate::gpu::uma::get().is_some_and(|u| u.contains(va)) || crate::hle::kernelx::in_graphics_reservation(va) {
        ResetWriteWatch(va as *const c_void, bytes);
    }
    let upload = (e.device, e.va, Arc::clone(&e.tex), e.state);
    TEXTURES.lock().unwrap().insert(res as usize, e);
    let r = ID3D12Resource::from_raw_borrowed(&res).ok_or("no resource")?;
    run_upload(upload, r)
}

/// The PC texture was released for good.
pub fn forget(res: usize) {
    TEXTURES.lock().unwrap().remove(&res);
}

/// The title-memory address of a texture kept in step with it.
pub fn address(res: usize) -> Option<usize> {
    TEXTURES.lock().unwrap().get(&res).map(|e| e.va)
}

/// Before the title's GPU work: textures whose memory the CPU wrote since the last look are
/// uploaded again.
pub(super) unsafe fn refresh(queue: *mut c_void) {
    thread_local! {
        static BUSY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    // the uploads' own ExecuteCommandLists come back here
    if BUSY.with(|b| b.replace(true)) {
        return;
    }
    refresh_once(queue);
    BUSY.with(|b| b.set(false));
}

unsafe fn refresh_once(queue: *mut c_void) {
    let Some(u) = crate::gpu::uma::get() else { return };
    let sparse = u.shader_windows() > u.windows();
    if TEXTURES.lock().unwrap().is_empty() && !sparse {
        return;
    }
    let mut dirty: Vec<usize> = Vec::new();
    let mut page = 0x1000u32;
    // the committed windows, the sparse ones (the title's reservations) and other reservations
    let regions: Vec<(usize, usize)> = (0..u.shader_windows()).map(|k| u.window(k)).chain(crate::hle::kernelx::graphics_reservations()).collect();
    for (base, size) in regions {
        loop {
            let mut buf = vec![std::ptr::null_mut::<c_void>(); 16384];
            let mut n = buf.len();
            if GetWriteWatch(WRITE_WATCH_FLAG_RESET, base as *const c_void, size, buf.as_mut_ptr(), &mut n, &mut page) != 0 {
                break;
            }
            dirty.extend(buf[..n].iter().map(|&p| p as usize));
            if n < buf.len() {
                break;
            }
        }
    }
    dirty.sort_unstable();
    // the title's own memory: what the CPU wrote goes to its video memory first
    if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) {
        super::sparse::before_execute(q, &dirty, page as usize);
    }
    if dirty.is_empty() {
        return;
    }
    // textures with a written page; a reference each while they are uploaded
    let todo: Vec<(ID3D12Resource, Upload)> = {
        let t = TEXTURES.lock().unwrap();
        t.iter()
            .filter(|(_, e)| {
                let i = dirty.partition_point(|&p| p + page as usize <= e.va);
                dirty.get(i).is_some_and(|&p| p < e.va + e.tex.size() as usize)
            })
            .filter_map(|(&res, e)| {
                let p = res as *mut c_void;
                ID3D12Resource::from_raw_borrowed(&p).map(|r| (r.clone(), (e.device, e.va, Arc::clone(&e.tex), e.state)))
            })
            .collect()
    };
    for (r, u) in todo {
        if let Err(e) = run_upload(u, &r) {
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("title texture: upload again failed: {e}");
            }
        }
    }
}

/// `len` bytes at `at`: the committed, readable pages copied, the rest zero (a streaming pool
/// has only some of its pages mapped).
unsafe fn read_committed(at: usize, len: usize) -> Vec<u8> {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    let mut out = vec![0u8; len];
    let mut p = at;
    while p < at + len {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(p as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 {
            break;
        }
        let end = (m.BaseAddress as usize + m.RegionSize).min(at + len);
        let readable = m.State == MEM_COMMIT && m.Protect & 0xEE != 0 && m.Protect & 0x100 == 0; // R / RW / WC / X-R…, no guard
        if readable {
            std::ptr::copy_nonoverlapping(p as *const u8, out.as_mut_ptr().add(p - at), end - p);
        }
        p = end;
    }
    out
}

/// Every subresource (D3D12 order) of a texture from the title's memory at `va`.
unsafe fn subresources(va: usize, tex: &xg::Texture) -> Vec<Subresource> {
    let Some(tex) = tex.texels() else { return Vec::new() };
    let c = &tex.chain;
    let bpe = tex.bpe as usize;
    let mut out = Vec::with_capacity(tex.array as usize * c.levels.len());
    for s in 0..tex.array as usize {
        let slice = read_committed(va + s * c.slice_bytes, c.slice_bytes);
        for (i, l) in c.levels.iter().enumerate() {
            out.push(Subresource { depth: 1, rows: l.height, row_bytes: l.width * bpe, data: c.detile_level(&slice, i, s) });
        }
    }
    out
}

unsafe fn run_upload((device, va, tex, state): Upload, res: &ID3D12Resource) -> Result<(), String> {
    let d = device as *mut c_void;
    let dev = ID3D12Device::from_raw_borrowed(&d).ok_or("no device")?;
    let queue = {
        let mut q = QUEUES.lock().unwrap();
        if let Some(q) = q.get(&device) {
            q.clone()
        } else {
            let made: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(|e| e.to_string())?;
            q.insert(device, made.clone());
            made
        }
    };
    let subs = subresources(va, &tex);
    texture::upload(dev, &queue, res, &subs, state, state)?;
    UPLOADS.fetch_add(1, Ordering::Relaxed);
    Ok(())
}
