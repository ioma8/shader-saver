import AppKit
import Metal
import MetalKit
import simd

// MARK: - Data structures

struct Vertex {
    var position: SIMD3<Float>
    var normal: SIMD3<Float>
}

struct Uniforms {
    var modelMatrix: float4x4
    var viewMatrix: float4x4
    var projectionMatrix: float4x4
    var cameraPosition: SIMD4<Float>  // w unused, float4 avoids Metal padding mismatch
}

// MARK: - STL Parser

func loadSTL(from url: URL) -> [Vertex] {
    guard let data = try? Data(contentsOf: url) else {
        print("Failed to load STL: \(url.path)"); return []
    }

    // ASCII STL starts with "solid" (binary can too but is caught by size check)
    let isASCII = data.count > 5 && data.prefix(5).elementsEqual("solid".utf8)
    if isASCII, let text = String(data: data, encoding: .utf8) {
        return parseASCIISTL(text)
    }
    return parseBinarySTL(data)
}

private func parseBinarySTL(_ data: Data) -> [Vertex] {
    guard data.count >= 84 else { return [] }

    let triangleCount = data.withUnsafeBytes {
        $0.loadUnaligned(fromByteOffset: 80, as: UInt32.self)
    }

    var vertices: [Vertex] = []
    vertices.reserveCapacity(Int(triangleCount) * 3)

    for i in 0..<Int(triangleCount) {
        let base = 84 + i * 50
        guard base + 50 <= data.count else { break }

        func vec3(at offset: Int) -> SIMD3<Float> {
            data.withUnsafeBytes {
                SIMD3<Float>(
                    $0.loadUnaligned(fromByteOffset: offset,     as: Float.self),
                    $0.loadUnaligned(fromByteOffset: offset + 4, as: Float.self),
                    $0.loadUnaligned(fromByteOffset: offset + 8, as: Float.self)
                )
            }
        }

        let normal = vec3(at: base)
        for v in 0..<3 {
            vertices.append(Vertex(position: vec3(at: base + 12 + v * 12), normal: normal))
        }
    }
    return vertices
}

private func parseASCIISTL(_ text: String) -> [Vertex] {
    var vertices: [Vertex] = []
    var currentNormal = SIMD3<Float>.zero
    var faceVerts: [SIMD3<Float>] = []

    for line in text.components(separatedBy: .newlines) {
        let parts = line.trimmingCharacters(in: .whitespaces)
            .components(separatedBy: .whitespaces)
            .filter { !$0.isEmpty }
        guard !parts.isEmpty else { continue }

        switch parts[0] {
        case "facet" where parts.count == 5:
            currentNormal = SIMD3<Float>(
                Float(parts[2]) ?? 0,
                Float(parts[3]) ?? 0,
                Float(parts[4]) ?? 0
            )
            faceVerts = []
        case "vertex" where parts.count == 4:
            faceVerts.append(SIMD3<Float>(
                Float(parts[1]) ?? 0,
                Float(parts[2]) ?? 0,
                Float(parts[3]) ?? 0
            ))
        case "endfacet":
            for pos in faceVerts {
                vertices.append(Vertex(position: pos, normal: currentNormal))
            }
        default:
            break
        }
    }
    return vertices
}

// MARK: - Matrix helpers

extension float4x4 {
    static func translation(_ t: SIMD3<Float>) -> float4x4 {
        var m = float4x4(1)
        m[3] = SIMD4<Float>(t.x, t.y, t.z, 1)
        return m
    }

    static func rotationY(_ a: Float) -> float4x4 {
        let c = cos(a), s = sin(a)
        return float4x4(
            SIMD4<Float>( c, 0, s, 0),
            SIMD4<Float>( 0, 1, 0, 0),
            SIMD4<Float>(-s, 0, c, 0),
            SIMD4<Float>( 0, 0, 0, 1)
        )
    }

    static func rotationX(_ a: Float) -> float4x4 {
        let c = cos(a), s = sin(a)
        return float4x4(
            SIMD4<Float>(1,  0,  0, 0),
            SIMD4<Float>(0,  c, -s, 0),
            SIMD4<Float>(0,  s,  c, 0),
            SIMD4<Float>(0,  0,  0, 1)
        )
    }

    static func perspective(fovY: Float, aspect: Float, near: Float, far: Float) -> float4x4 {
        let y = 1 / tan(fovY * 0.5)
        let z = far / (near - far)
        return float4x4(
            SIMD4<Float>(y / aspect, 0,  0,        0),
            SIMD4<Float>(0,          y,  0,        0),
            SIMD4<Float>(0,          0,  z,       -1),
            SIMD4<Float>(0,          0,  z * near, 0)
        )
    }

    static func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> float4x4 {
        let f = normalize(center - eye)
        let r = normalize(cross(f, up))
        let u = cross(r, f)
        return float4x4(
            SIMD4<Float>(r.x, u.x, -f.x, 0),
            SIMD4<Float>(r.y, u.y, -f.y, 0),
            SIMD4<Float>(r.z, u.z, -f.z, 0),
            SIMD4<Float>(-dot(r, eye), -dot(u, eye), dot(f, eye), 1)
        )
    }
}

// MARK: - Renderer

final class STLRenderer: NSObject, MTKViewDelegate {
    private let queue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private let depthState: MTLDepthStencilState
    private let vertexBuffer: MTLBuffer
    private let vertexCount: Int
    private let modelCenter: SIMD3<Float>
    private let modelScale: Float

    var rotationY: Float = Float.pi + 0.5
    var rotationX: Float = 0.0
    var cameraDistance: Float = 3.0

    init(device: MTLDevice, vertices: [Vertex]) {
        self.queue = device.makeCommandQueue()!
        self.vertexCount = vertices.count

        self.vertexBuffer = device.makeBuffer(
            bytes: vertices,
            length: vertices.count * MemoryLayout<Vertex>.stride,
            options: .storageModeShared
        )!

        var minP = SIMD3<Float>(repeating:  Float.greatestFiniteMagnitude)
        var maxP = SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude)
        for v in vertices {
            minP = min(minP, v.position)
            maxP = max(maxP, v.position)
        }
        self.modelCenter = (minP + maxP) * 0.5
        let size = simd_length(maxP - minP)
        self.modelScale = size > 0 ? 2.0 / size : 1.0

        let library = try! device.makeLibrary(source: STLRenderer.shaderSource, options: nil)
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = library.makeFunction(name: "vertexMain")
        desc.fragmentFunction = library.makeFunction(name: "fragmentMain")
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        desc.depthAttachmentPixelFormat = .depth32Float
        self.pipeline = try! device.makeRenderPipelineState(descriptor: desc)

        let ds = MTLDepthStencilDescriptor()
        ds.depthCompareFunction = .less
        ds.isDepthWriteEnabled = true
        self.depthState = device.makeDepthStencilState(descriptor: ds)!

        super.init()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard
            let pass = view.currentRenderPassDescriptor,
            let drawable = view.currentDrawable,
            let commandBuffer = queue.makeCommandBuffer(),
            let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass)
        else { return }

        let size = view.drawableSize
        let aspect = Float(size.width / size.height)

        let center  = float4x4.translation(-modelCenter)
        let scale   = float4x4(diagonal: SIMD4<Float>(modelScale, modelScale, modelScale, 1))
        // STL files are Z-up (CAD convention); rotate to Y-up before applying user rotation
        let zToY    = float4x4.rotationX(.pi / 2)
        let model   = float4x4.rotationX(rotationX) * float4x4.rotationY(rotationY) * zToY * scale * center

        let eye = SIMD3<Float>(0, 0, -cameraDistance)
        var uniforms = Uniforms(
            modelMatrix: model,
            viewMatrix: float4x4.lookAt(
                eye:    eye,
                center: SIMD3<Float>(0, 0, 0),
                up:     SIMD3<Float>(0, 1, 0)
            ),
            projectionMatrix: float4x4.perspective(
                fovY: Float.pi / 3,
                aspect: aspect,
                near: 0.01,
                far: 100
            ),
            cameraPosition: SIMD4<Float>(eye.x, eye.y, eye.z, 0)
        )

        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthState)
        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: vertexCount)
        encoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    static let shaderSource = #"""
    #include <metal_stdlib>
    using namespace metal;

    struct Vertex {
        float3 position;
        float3 normal;
    };

    struct Uniforms {
        float4x4 modelMatrix;
        float4x4 viewMatrix;
        float4x4 projectionMatrix;
        float4 cameraPosition;
    };

    struct VertexOut {
        float4 position [[position]];
        float3 worldNormal;
        float3 worldPos;
    };

    vertex VertexOut vertexMain(
        uint vertexID [[vertex_id]],
        constant Vertex* vertices [[buffer(0)]],
        constant Uniforms& uniforms [[buffer(1)]]
    ) {
        Vertex v = vertices[vertexID];
        float4 worldPos = uniforms.modelMatrix * float4(v.position, 1.0);
        VertexOut out;
        out.position    = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos;
        out.worldNormal = (uniforms.modelMatrix * float4(v.normal, 0.0)).xyz;
        out.worldPos    = worldPos.xyz;
        return out;
    }

    fragment float4 fragmentMain(
        VertexOut in [[stage_in]],
        constant Uniforms& uniforms [[buffer(1)]]
    ) {
        float3 n        = normalize(in.worldNormal);
        float3 lightDir = normalize(float3(1.0, 2.0, -1.0));
        float  diffuse  = max(0.0, dot(n, lightDir));
        float  backFill = max(0.0, dot(-n, lightDir)) * 0.2;
        float3 color    = float3(0.8, 0.7, 0.6) * (diffuse + 0.15 + backFill);

        float3 viewDir  = normalize(uniforms.cameraPosition.xyz - in.worldPos);
        float  rim      = 1.0 - max(0.0, dot(n, viewDir));
        rim             = pow(rim, 2.5);
        color          += float3(0.2, 0.5, 1.0) * rim * 1.8;

        return float4(color, 1.0);
    }
    """#
}

// MARK: - View

final class STLView: MTKView {
    var renderer: STLRenderer?
    private var lastDrag: NSPoint?

    override var acceptsFirstResponder: Bool { true }

    override func mouseDown(with event: NSEvent) {
        lastDrag = convert(event.locationInWindow, from: nil)
    }

    override func mouseDragged(with event: NSEvent) {
        let pos = convert(event.locationInWindow, from: nil)
        guard let last = lastDrag else { return }
        renderer?.rotationY += Float(pos.x - last.x) * 0.01
        renderer?.rotationX += Float(pos.y - last.y) * 0.01
        lastDrag = pos
    }

    override func mouseUp(with event: NSEvent) { lastDrag = nil }

    override func scrollWheel(with event: NSEvent) {
        guard let renderer else { return }
        renderer.cameraDistance = max(0.5, min(20.0, renderer.cameraDistance - Float(event.deltaY) * 0.1))
    }

    override func keyDown(with event: NSEvent) {
        if event.keyCode == 53 { NSApp.terminate(nil) }
        else { super.keyDown(with: event) }
    }
}

// MARK: - App entry point

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow?
    private var device: MTLDevice?
    private var pendingURL: URL?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let menu = NSMenu()
        let appMenuItem = NSMenuItem()
        menu.addItem(appMenuItem)
        let appMenu = NSMenu()
        appMenu.addItem(NSMenuItem(title: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
        appMenuItem.submenu = appMenu
        NSApp.mainMenu = menu

        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal not supported"); NSApp.terminate(nil); return
        }
        self.device = device

        if let url = pendingURL {
            open(url: url)
        } else if let path = CommandLine.arguments.dropFirst().first {
            open(url: URL(fileURLWithPath: path))
        } else {
            let panel = NSOpenPanel()
            panel.title = "Open STL File"
            panel.allowedContentTypes = [.init(filenameExtension: "stl")!]
            guard panel.runModal() == .OK, let url = panel.url else {
                NSApp.terminate(nil); return
            }
            open(url: url)
        }
    }

    // Called by macOS when a file is opened via Finder double-click or "Open With"
    func application(_ sender: NSApplication, openFile filename: String) -> Bool {
        let url = URL(fileURLWithPath: filename)
        if device != nil { open(url: url) } else { pendingURL = url }
        return true
    }

    private func open(url: URL) {
        guard let device else { return }

        let vertices = loadSTL(from: url)
        guard !vertices.isEmpty else {
            print("No vertices loaded from \(url.lastPathComponent)"); NSApp.terminate(nil); return
        }
        print("Loaded \(vertices.count / 3) triangles from \(url.lastPathComponent)")

        let renderer = STLRenderer(device: device, vertices: vertices)
        let view = STLView(frame: NSRect(x: 0, y: 0, width: 800, height: 600), device: device)
        view.delegate = renderer
        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        view.clearColor = MTLClearColorMake(0.1, 0.1, 0.12, 1.0)
        view.preferredFramesPerSecond = 60
        view.renderer = renderer

        let win = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 800, height: 600),
            styleMask: [.titled, .closable, .resizable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        win.title = url.lastPathComponent
        win.contentView = view
        win.center()
        win.makeKeyAndOrderFront(nil)
        win.makeFirstResponder(view)
        NSApp.activate(ignoringOtherApps: true)
        self.window = win
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

@main
enum STLViewer {
    @MainActor
    static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.setActivationPolicy(.regular)
        app.delegate = delegate
        app.run()
    }
}
