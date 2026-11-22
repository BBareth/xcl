//! `minuserx.dll` / the console's ntuser API sets: the window functions titles use; the rest
//! is Windows' user32.
//!
//! Console titles create their window with the ANSI functions (CreateWindowExA,
//! RegisterClassExA) but give it DefWindowProcW (XDefiant): on Windows that makes the title
//! text show as CJK characters (8-bit text read as UTF-16). The console's user32 is Unicode
//! underneath, so the ANSI calls are made as Unicode ones here – the window is a Unicode window
//! with the text converted.

#![allow(non_snake_case)]

use std::ffi::c_void;

use crate::util::{from_ansi_ptr, wide};

#[link(name = "user32")]
extern "system" {
    fn RegisterClassExW(c: *const WndClassEx<u16>) -> u16;
    #[allow(clippy::too_many_arguments)]
    fn CreateWindowExW(ex_style: u32, class: *const u16, title: *const u16, style: u32, x: i32, y: i32, w: i32, h: i32, parent: *mut c_void, menu: *mut c_void, instance: *mut c_void, param: *const c_void) -> *mut c_void;
}

pub fn exports() -> Vec<(&'static str, usize)> {
    crate::hle_exports![CreateWindowExA, RegisterClassExA]
}

/// WNDCLASSEXA (`u8`) / WNDCLASSEXW (`u16`): the same layout, the strings are pointers
#[repr(C)]
pub struct WndClassEx<C> {
    cb_size: u32,
    style: u32,
    wnd_proc: usize,
    cls_extra: i32,
    wnd_extra: i32,
    instance: usize,
    icon: usize,
    cursor: usize,
    background: usize,
    menu_name: *const C,
    class_name: *const C,
    icon_small: usize,
}

/// A class / menu name: an atom (below 0x10000) as it is, a string as UTF-16 (kept alive in
/// `keep`).
unsafe fn name(p: *const u8, keep: &mut Vec<Vec<u16>>) -> *const u16 {
    if (p as usize) < 0x1_0000 {
        return p as *const u16;
    }
    keep.push(wide(&from_ansi_ptr(p).unwrap_or_default()));
    keep.last().unwrap().as_ptr()
}

pub unsafe extern "system" fn RegisterClassExA(c: *const WndClassEx<u8>) -> u16 {
    if c.is_null() {
        return 0;
    }
    let c = &*c;
    let mut keep = Vec::new();
    let w = WndClassEx::<u16> {
        cb_size: std::mem::size_of::<WndClassEx<u16>>() as u32,
        style: c.style,
        wnd_proc: c.wnd_proc,
        cls_extra: c.cls_extra,
        wnd_extra: c.wnd_extra,
        instance: c.instance,
        icon: c.icon,
        cursor: c.cursor,
        background: c.background,
        menu_name: name(c.menu_name, &mut keep),
        class_name: name(c.class_name, &mut keep),
        icon_small: c.icon_small,
    };
    RegisterClassExW(&w)
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "system" fn CreateWindowExA(ex_style: u32, class: *const u8, title: *const u8, style: u32, x: i32, y: i32, w: i32, h: i32, parent: *mut c_void, menu: *mut c_void, instance: *mut c_void, param: *const c_void) -> *mut c_void {
    let mut keep = Vec::new();
    let class = name(class, &mut keep);
    let title = if title.is_null() { std::ptr::null() } else { name(title, &mut keep) };
    CreateWindowExW(ex_style, class, title, style, x, y, w, h, parent, menu, instance, param)
}
