#include <windows.h>
#include <dxgi1_2.h>
#include <cstdio>
int main() {
  IDXGIFactory1* factory = nullptr;
  if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory))) return 1;
  for (UINT i = 0;; ++i) {
    IDXGIAdapter1* adapter = nullptr;
    if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) break;
    DXGI_ADAPTER_DESC1 d{};
    if (FAILED(adapter->GetDesc1(&d))) return 2;
    wprintf(L"device_id=%u name=%ls vendor=%04x device=%04x luid=%08x:%08x flags=%u\n", i, d.Description, d.VendorId, d.DeviceId, (unsigned)d.AdapterLuid.HighPart, d.AdapterLuid.LowPart, d.Flags);
    adapter->Release();
  }
  factory->Release();
}
