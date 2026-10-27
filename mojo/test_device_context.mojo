from std.gpu.host import DeviceContext

def main() raises:
    print("Testing Mojo DeviceContext...")
    var ctx = DeviceContext(0)
    print("DeviceContext created successfully: " + ctx.name())
