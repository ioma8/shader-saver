import AppKit
import Metal
import MetalKit
import simd

private let renderScale: CGFloat = 0.6

private struct Uniforms {
    var resolution: SIMD2<Float>
    var time: Float
}

func scaledDrawableSize(bounds: CGRect, scaleFactor: CGFloat) -> CGSize {
    CGSize(
        width: max(1, bounds.width * scaleFactor * renderScale),
        height: max(1, bounds.height * scaleFactor * renderScale)
    )
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var window: FullscreenWindow?
    private var renderer: Renderer?

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard
            let screen = NSScreen.main,
            let device = MTLCreateSystemDefaultDevice()
        else {
            NSApp.terminate(nil)
            return
        }

        let renderer = Renderer(device: device)
        let view = ShaderView(frame: screen.frame, device: device, renderer: renderer)
        let window = FullscreenWindow(contentRect: screen.frame)
        window.contentView = view
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        NSCursor.hide()
        self.renderer = renderer
        self.window = window
    }

    func applicationWillTerminate(_ notification: Notification) {
        NSCursor.unhide()
    }
}

final class FullscreenWindow: NSWindow {
    init(contentRect: NSRect) {
        super.init(
            contentRect: contentRect,
            styleMask: .borderless,
            backing: .buffered,
            defer: false
        )

        level = .screenSaver
        collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary]
        backgroundColor = .black
        isOpaque = true
    }

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
}

final class ShaderView: MTKView {
    init(frame: NSRect, device: MTLDevice, renderer: Renderer) {
        super.init(frame: frame, device: device)
        self.device = device
        self.delegate = renderer
        self.clearColor = MTLClearColorMake(0, 0, 0, 1)
        self.colorPixelFormat = .bgra8Unorm
        self.framebufferOnly = false
        self.isPaused = false
        self.enableSetNeedsDisplay = false
        self.preferredFramesPerSecond = 60
        self.autoResizeDrawable = false
        updateDrawableSize()
    }

    @available(*, unavailable)
    required init(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override var acceptsFirstResponder: Bool { true }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        window?.makeFirstResponder(self)
        updateDrawableSize()
    }

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        updateDrawableSize()
    }

    override func keyDown(with event: NSEvent) {
        if event.keyCode == 53 {
            NSApp.terminate(nil)
            return
        }
        super.keyDown(with: event)
    }

    private func updateDrawableSize() {
        let scaleFactor = window?.screen?.backingScaleFactor ?? NSScreen.main?.backingScaleFactor ?? 2.0
        drawableSize = scaledDrawableSize(bounds: bounds, scaleFactor: scaleFactor)
    }
}

final class Renderer: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private let start = CACurrentMediaTime()

    init(device: MTLDevice) {
        self.device = device
        self.queue = device.makeCommandQueue()!

        let library = try! device.makeLibrary(source: ShaderSource.metal, options: nil)
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = library.makeFunction(name: "vertexMain")
        descriptor.fragmentFunction = library.makeFunction(name: "fragmentMain")
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        self.pipeline = try! device.makeRenderPipelineState(descriptor: descriptor)
        super.init()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard
            let pass = view.currentRenderPassDescriptor,
            let drawable = view.currentDrawable,
            let commandBuffer = queue.makeCommandBuffer(),
            let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass)
        else {
            return
        }

        var uniforms = Uniforms(
            resolution: SIMD2(Float(view.drawableSize.width), Float(view.drawableSize.height)),
            time: Float(CACurrentMediaTime() - start)
        )

        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}

enum ShaderSource {
    static let metal = #"""
    #include <metal_stdlib>
    using namespace metal;

    struct Uniforms {
        float2 resolution;
        float time;
    };

    struct VertexOut {
        float4 position [[position]];
    };

    vertex VertexOut vertexMain(uint vertexID [[vertex_id]]) {
        float2 positions[3] = {
            float2(-1.0, -1.0),
            float2( 3.0, -1.0),
            float2(-1.0,  3.0)
        };

        VertexOut out;
        out.position = float4(positions[vertexID], 0.0, 1.0);
        return out;
    }

    fragment float4 fragmentMain(
        VertexOut in [[stage_in]],
        constant Uniforms& uniforms [[buffer(0)]]
    ) {
        float2 r = uniforms.resolution;
        float t = uniforms.time * 0.5;
        float2 fragCoord = in.position.xy;
        float4 FC = float4(fragCoord, 0.5, 1.0);
        float4 o = float4(0.0);

        for (float i = 0.0, z = 0.0, d = 0.0, s = 0.0; i++ < 1e2;) {
            float3 v;
            float3 p = z * normalize(FC.rgb * 2.0 - float3(r.x, r.y, r.y));
            p.z += 2.0;
            float4 m = cos(p.y + t + float4(0.0, 11.0, 33.0, 0.0));
            float2 zx = p.zx * float2x2(m.x, m.y, m.z, m.w);
            p.zx = zx;
            v = p;
            for (d = 1.0; d < i; d += d) {
                p += 1.08 * sin(p.yzx * d + t * 0.2) / d;
            }
            z += d = 0.2 * max(0.03 + abs((s = cos(3.0 * p.y))) * 0.1, length(v) - 1.0);
            o += (cos(s / 0.4 + p.y + float4(6.0, 1.0, 3.0, 0.0)) + 1.5) / d / z;
        }

        o = tanh(o / 1e4);
        o.a = 1.0;
        return o;
    }
    """#
}

@main
enum ShaderSaver {
    @MainActor
    static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.setActivationPolicy(.regular)
        app.delegate = delegate
        app.run()
    }
}
