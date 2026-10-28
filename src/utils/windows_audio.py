import sys
import os

def is_windows() -> bool:
    return sys.platform.startswith('win')

def ensure_microphone_unmuted() -> dict:
    result = {'supported': False, 'is_muted': False, 'volume_percent': 100.0, 'unmuted_action_taken': False, 'error': None}
    if not is_windows():
        result['supported'] = False
        return result
    try:
        import ctypes
        from ctypes import wintypes, POINTER, c_void_p, Structure, c_int, byref, c_float, c_uint32, cast
        import uuid
        ole32 = ctypes.windll.ole32

        class GUID(Structure):
            _fields_ = [('Data1', c_uint32), ('Data2', wintypes.WORD), ('Data3', wintypes.WORD), ('Data4', wintypes.BYTE * 8)]

            def __init__(self, guid_str):
                super().__init__()
                u = uuid.UUID(guid_str)
                self.Data1 = u.time_low
                self.Data2 = u.time_mid
                self.Data3 = u.time_hi_version
                for i, b in enumerate(u.bytes[8:]):
                    self.Data4[i] = b
        ole32.CoInitialize(None)
        CLSID_MMDeviceEnumerator = GUID('{BCDE0395-E52F-467C-8E3D-C4579291692E}')
        IID_IMMDeviceEnumerator = GUID('{A95664D2-9614-4F35-A746-DE8DB63617E6}')
        IID_IAudioEndpointVolume = GUID('{5CDF2C82-841E-4546-9722-0CF74078229A}')
        pEnum = c_void_p()
        hr = ole32.CoCreateInstance(byref(CLSID_MMDeviceEnumerator), None, 1 | 2 | 4 | 16, byref(IID_IMMDeviceEnumerator), byref(pEnum))
        if hr != 0 or not pEnum.value:
            result['error'] = f'CoCreateInstance failed (hr={hr})'
            return result
        vtable_ptr = cast(pEnum.value, POINTER(POINTER(c_void_p))).contents
        GetDefaultAudioEndpoint_proto = ctypes.WINFUNCTYPE(c_int, c_void_p, c_int, c_int, POINTER(c_void_p))
        GetDefaultAudioEndpoint = GetDefaultAudioEndpoint_proto(vtable_ptr[4])
        pDevice = c_void_p()
        hr2 = GetDefaultAudioEndpoint(pEnum, 1, 0, byref(pDevice))
        if hr2 != 0 or not pDevice.value:
            result['error'] = f'GetDefaultAudioEndpoint failed (hr={hr2})'
            return result
        dev_vtable = cast(pDevice.value, POINTER(POINTER(c_void_p))).contents
        Activate_proto = ctypes.WINFUNCTYPE(c_int, c_void_p, POINTER(GUID), c_uint32, c_void_p, POINTER(c_void_p))
        Activate = Activate_proto(dev_vtable[3])
        pVolume = c_void_p()
        hr3 = Activate(pDevice, byref(IID_IAudioEndpointVolume), 1 | 2 | 4 | 16, None, byref(pVolume))
        if hr3 != 0 or not pVolume.value:
            result['error'] = f'Activate IAudioEndpointVolume failed (hr={hr3})'
            return result
        vol_vtable = cast(pVolume.value, POINTER(POINTER(c_void_p))).contents
        GetMute_proto = ctypes.WINFUNCTYPE(c_int, c_void_p, POINTER(wintypes.BOOL))
        GetMute = GetMute_proto(vol_vtable[15])
        is_muted = wintypes.BOOL()
        GetMute(pVolume, byref(is_muted))
        GetVol_proto = ctypes.WINFUNCTYPE(c_int, c_void_p, POINTER(c_float))
        GetVol = GetVol_proto(vol_vtable[9])
        vol = c_float()
        GetVol(pVolume, byref(vol))
        was_muted = bool(is_muted.value)
        cur_vol = vol.value * 100.0
        result['supported'] = True
        result['is_muted'] = was_muted
        result['volume_percent'] = round(cur_vol, 1)
        if was_muted:
            SetMute_proto = ctypes.WINFUNCTYPE(c_int, c_void_p, wintypes.BOOL, c_void_p)
            SetMute = SetMute_proto(vol_vtable[14])
            SetMute(pVolume, wintypes.BOOL(False), None)
            result['unmuted_action_taken'] = True
            result['is_muted'] = False
            print('[AUDIO-SYS] Detected muted Windows microphone. Successfully UNMUTED system mic!')
        if cur_vol < 50.0:
            SetVol_proto = ctypes.WINFUNCTYPE(c_int, c_void_p, c_float, c_void_p)
            SetVol = SetVol_proto(vol_vtable[7])
            SetVol(pVolume, c_float(0.85), None)
            result['volume_percent'] = 85.0
            print(f'[AUDIO-SYS] Microphone volume was low ({cur_vol:.1f}%). Boosted to 85.0%!')
        return result
    except Exception as e:
        result['error'] = str(e)
        return result
if __name__ == '__main__':
    st = ensure_microphone_unmuted()
    print('Microphone Status:', st)
