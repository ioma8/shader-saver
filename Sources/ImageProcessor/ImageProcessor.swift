import AppKit
import Metal
import MetalKit
import UniformTypeIdentifiers

// MARK: - Settings

struct ProcessorSettings {
    var contrast: Float = 1.0       // 1.0 = unchanged, <1 = flat, >1 = punchy
    var blurRadius: Int32 = 0       // pixels
    var unsharpStrength: Float = 0  // 0 = off
}

// MARK: - GPU Processor

@MainActor
final class GPUProcessor: NSObject, MTKViewDelegate {
    let device: MTLDevice
    private let queue: MTLCommandQueue
    private let contrastPipeline: MTLComputePipelineState
    private let blurPipeline: MTLComputePipelineState
    private let sharpenPipeline: MTLComputePipelineState
    private let displayPipeline: MTLRenderPipelineState

    private var inputTexture: MTLTexture?
    private var tex1: MTLTexture?        // post-contrast
    private var tex2: MTLTexture?        // post-sharpen
    private var outputTexture: MTLTexture?
    private var imageAspect: Float = 1.0

    var settings = ProcessorSettings()
    var hasImage: Bool { inputTexture != nil }

    init(device: MTLDevice) {
        self.device = device
        self.queue  = device.makeCommandQueue()!

        let library = try! device.makeLibrary(source: GPUProcessor.shaderSource, options: nil)

        contrastPipeline = try! device.makeComputePipelineState(
            function: library.makeFunction(name: "contrastKernel")!)
        blurPipeline     = try! device.makeComputePipelineState(
            function: library.makeFunction(name: "boxBlurKernel")!)
        sharpenPipeline  = try! device.makeComputePipelineState(
            function: library.makeFunction(name: "sharpenKernel")!)

        let pd = MTLRenderPipelineDescriptor()
        pd.vertexFunction   = library.makeFunction(name: "passthroughVertex")
        pd.fragmentFunction = library.makeFunction(name: "passthroughFragment")
        pd.colorAttachments[0].pixelFormat = .bgra8Unorm
        displayPipeline = try! device.makeRenderPipelineState(descriptor: pd)

        super.init()
    }

    // MARK: - Load

    func load(url: URL) {
        let loader = MTKTextureLoader(device: device)
        let opts: [MTKTextureLoader.Option: Any] = [
            .SRGB: false,
            .textureUsage: MTLTextureUsage.shaderRead.rawValue
        ]
        guard let tex = try? loader.newTexture(URL: url, options: opts) else { return }
        inputTexture = tex
        imageAspect  = Float(tex.width) / Float(tex.height)
        makeIntermediates(width: tex.width, height: tex.height)
        process()
    }

    private func makeIntermediates(width: Int, height: Int) {
        let d = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
        d.usage       = [.shaderRead, .shaderWrite]
        d.storageMode = .private
        tex1          = device.makeTexture(descriptor: d)
        tex2          = device.makeTexture(descriptor: d)
        outputTexture = device.makeTexture(descriptor: d)
    }

    // MARK: - Process

    func process() {
        guard
            let input = inputTexture,
            let t1 = tex1, let t2 = tex2, let output = outputTexture,
            let cb = queue.makeCommandBuffer()
        else { return }

        let w = input.width, h = input.height
        let tgSize  = MTLSize(width: 8, height: 8, depth: 1)
        let tgCount = MTLSize(width: (w + 7) / 8, height: (h + 7) / 8, depth: 1)

        func computePass(_ pipeline: MTLComputePipelineState, _ configure: (MTLComputeCommandEncoder) -> Void) {
            guard let enc = cb.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(pipeline)
            configure(enc)
            enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
            enc.endEncoding()
        }

        var contrast = settings.contrast
        computePass(contrastPipeline) {
            $0.setTexture(input, index: 0); $0.setTexture(t1, index: 1)
            $0.setBytes(&contrast, length: 4, index: 0)
        }

        // Pass 2: sharpen — reads clean contrast image so unsharp mask anchors off original signal
        var strength = settings.unsharpStrength
        computePass(sharpenPipeline) {
            $0.setTexture(t1, index: 0); $0.setTexture(t2, index: 1)
            $0.setBytes(&strength, length: 4, index: 0)
        }

        // Pass 3: box blur applied after sharpening
        var radius = settings.blurRadius
        computePass(blurPipeline) {
            $0.setTexture(t2, index: 0); $0.setTexture(output, index: 1)
            $0.setBytes(&radius, length: 4, index: 0)
        }

        cb.commit()
        cb.waitUntilCompleted()
    }

    // MARK: - Export

    func export(to url: URL) {
        guard let output = outputTexture else { return }

        // Private textures can't be read on CPU — blit to shared staging first
        let sd = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: output.width, height: output.height, mipmapped: false)
        sd.storageMode = .shared
        guard
            let staging = device.makeTexture(descriptor: sd),
            let cb   = queue.makeCommandBuffer(),
            let blit = cb.makeBlitCommandEncoder()
        else { return }

        blit.copy(
            from: output,   sourceSlice: 0, sourceLevel: 0,
            sourceOrigin:   MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize:     MTLSize(width: output.width, height: output.height, depth: 1),
            to: staging,    destinationSlice: 0, destinationLevel: 0,
            destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let w = staging.width, h = staging.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        staging.getBytes(&bytes, bytesPerRow: w * 4,
                         from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)

        guard
            let provider = CGDataProvider(data: Data(bytes) as CFData),
            let cgImage  = CGImage(
                width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { return }

        let rep = NSBitmapImageRep(cgImage: cgImage)
        if let png = rep.representation(using: .png, properties: [:]) {
            try? png.write(to: url)
        }
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard
            let output   = outputTexture,
            let pass     = view.currentRenderPassDescriptor,
            let drawable = view.currentDrawable,
            let cb       = queue.makeCommandBuffer(),
            let enc      = cb.makeRenderCommandEncoder(descriptor: pass)
        else { return }

        // Scale UV to maintain aspect ratio (letterbox)
        let viewAspect = Float(view.drawableSize.width) / Float(view.drawableSize.height)
        var scale = SIMD2<Float>(1, 1)
        if viewAspect > imageAspect {
            scale.x = viewAspect / imageAspect
        } else {
            scale.y = imageAspect / viewAspect
        }

        enc.setRenderPipelineState(displayPipeline)
        enc.setFragmentTexture(output, index: 0)
        enc.setFragmentBytes(&scale, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()

        cb.present(drawable)
        cb.commit()
    }

    // MARK: - Shaders

    static let shaderSource = #"""
    #include <metal_stdlib>
    using namespace metal;

    // --- Compute kernels ---

    kernel void contrastKernel(
        texture2d<float, access::read>  input    [[texture(0)]],
        texture2d<float, access::write> output   [[texture(1)]],
        constant float&                 contrast [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        if (gid.x >= input.get_width() || gid.y >= input.get_height()) return;
        float4 c = input.read(gid);
        c.rgb = clamp((c.rgb - 0.5) * contrast + 0.5, 0.0, 1.0);
        output.write(c, gid);
    }

    kernel void boxBlurKernel(
        texture2d<float, access::read>  input  [[texture(0)]],
        texture2d<float, access::write> output [[texture(1)]],
        constant int&                   radius [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        if (gid.x >= input.get_width() || gid.y >= input.get_height()) return;
        if (radius == 0) { output.write(input.read(gid), gid); return; }
        float4 sum   = float4(0);
        float  count = 0;
        for (int x = -radius; x <= radius; x++) {
            for (int y = -radius; y <= radius; y++) {
                uint2 coord = uint2(
                    clamp((int)gid.x + x, 0, (int)input.get_width()  - 1),
                    clamp((int)gid.y + y, 0, (int)input.get_height() - 1));
                sum   += input.read(coord);
                count += 1.0;
            }
        }
        output.write(sum / count, gid);
    }

    // Sharpening via unsharp mask — does its own inline 5x5 blur for the mask,
    // so it's independent of the blur slider.
    kernel void sharpenKernel(
        texture2d<float, access::read>  input    [[texture(0)]],
        texture2d<float, access::write> output   [[texture(1)]],
        constant float&                 strength [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        if (gid.x >= input.get_width() || gid.y >= input.get_height()) return;
        float4 orig = input.read(gid);
        if (strength == 0.0) { output.write(orig, gid); return; }
        float4 blur  = float4(0);
        float  count = 0;
        for (int x = -2; x <= 2; x++) {
            for (int y = -2; y <= 2; y++) {
                uint2 coord = uint2(
                    clamp((int)gid.x + x, 0, (int)input.get_width()  - 1),
                    clamp((int)gid.y + y, 0, (int)input.get_height() - 1));
                blur  += input.read(coord);
                count += 1.0;
            }
        }
        blur /= count;
        output.write(clamp(orig + (orig - blur) * strength, 0.0, 1.0), gid);
    }

    // --- Display ---

    struct VertexOut {
        float4 position [[position]];
        float2 uv;
    };

    // Fullscreen triangle — same trick as ShaderSaver
    vertex VertexOut passthroughVertex(uint vid [[vertex_id]]) {
        float2 pos[3] = { float2(-1,-1), float2(3,-1), float2(-1, 3) };
        float2 uvs[3] = { float2( 0, 1), float2(2, 1), float2( 0,-1) };
        VertexOut out;
        out.position = float4(pos[vid], 0, 1);
        out.uv       = uvs[vid];
        return out;
    }

    // scale is (viewAspect/imgAspect, 1) or (1, imgAspect/viewAspect) — whichever letterboxes
    fragment float4 passthroughFragment(
        VertexOut        in    [[stage_in]],
        texture2d<float> tex   [[texture(0)]],
        constant float2& scale [[buffer(0)]]
    ) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 uv = (in.uv - 0.5) * scale + 0.5;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
            return float4(0.12, 0.12, 0.12, 1.0);
        return tex.sample(s, uv);
    }
    """#
}

// MARK: - Preview view (drag-drop enabled MTKView)

final class PreviewView: MTKView {
    var onDrop: ((URL) -> Void)?

    override init(frame: CGRect, device: (any MTLDevice)?) {
        super.init(frame: frame, device: device)
        registerForDraggedTypes([.fileURL])
    }

    @available(*, unavailable)
    required init(coder: NSCoder) { fatalError() }

    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation { .copy }

    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        guard
            let items = sender.draggingPasteboard.readObjects(forClasses: [NSURL.self]) as? [URL],
            let url   = items.first
        else { return false }
        let ext = url.pathExtension.lowercased()
        guard ["png","jpg","jpeg","tiff","tif","heic","bmp"].contains(ext) else { return false }
        onDrop?(url)
        return true
    }
}

// MARK: - App

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow?
    private var processor: GPUProcessor?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Menu (needed for Cmd+Q)
        let menu = NSMenu()
        let item = NSMenuItem()
        menu.addItem(item)
        let appMenu = NSMenu()
        appMenu.addItem(NSMenuItem(title: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
        item.submenu = appMenu
        NSApp.mainMenu = menu

        guard let device = MTLCreateSystemDefaultDevice() else { NSApp.terminate(nil); return }

        let proc = GPUProcessor(device: device)
        self.processor = proc

        // Left: preview
        let preview = PreviewView(frame: NSRect(x: 0, y: 0, width: 720, height: 600), device: device)
        preview.delegate          = proc
        preview.colorPixelFormat  = .bgra8Unorm
        preview.clearColor        = MTLClearColorMake(0.12, 0.12, 0.12, 1)
        preview.isPaused          = false
        preview.preferredFramesPerSecond = 60
        preview.onDrop = { [weak self] url in self?.load(url: url) }

        // Right: controls
        let controls = buildControlsPanel()

        // Window with horizontal split
        let contentView = NSView()
        preview.translatesAutoresizingMaskIntoConstraints  = false
        controls.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(preview)
        contentView.addSubview(controls)

        NSLayoutConstraint.activate([
            // Controls: fixed right side
            controls.topAnchor.constraint(equalTo: contentView.topAnchor),
            controls.bottomAnchor.constraint(equalTo: contentView.bottomAnchor),
            controls.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            controls.widthAnchor.constraint(equalToConstant: 260),
            // Preview: fills rest
            preview.topAnchor.constraint(equalTo: contentView.topAnchor),
            preview.bottomAnchor.constraint(equalTo: contentView.bottomAnchor),
            preview.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
            preview.trailingAnchor.constraint(equalTo: controls.leadingAnchor),
        ])

        let win = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 980, height: 600),
            styleMask: [.titled, .closable, .resizable, .miniaturizable],
            backing: .buffered, defer: false)
        win.title       = "Image Processor"
        win.contentView = contentView
        win.contentMinSize = NSSize(width: 600, height: 400)
        win.center()
        win.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        self.window = win
    }

    private func load(url: URL) {
        processor?.load(url: url)
        window?.title = url.lastPathComponent
    }

    // MARK: - Controls panel

    private func buildControlsPanel() -> NSView {
        let panel = NSView()
        panel.wantsLayer = true
        panel.layer?.backgroundColor = NSColor(white: 0.16, alpha: 1).cgColor

        let stack = NSStackView()
        stack.orientation = .vertical
        stack.alignment   = .left
        stack.spacing     = 10
        stack.edgeInsets  = NSEdgeInsets(top: 24, left: 16, bottom: 24, right: 16)
        stack.translatesAutoresizingMaskIntoConstraints = false

        let sliderWidth: CGFloat = 228

        func heading(_ text: String) -> NSTextField {
            let f = NSTextField(labelWithString: text)
            f.font      = .systemFont(ofSize: 10, weight: .semibold)
            f.textColor = .secondaryLabelColor
            return f
        }

        func row(_ text: String) -> NSTextField {
            let f = NSTextField(labelWithString: text)
            f.font      = .systemFont(ofSize: 12)
            f.textColor = .labelColor
            return f
        }

        func slider(min: Double, max: Double, value: Double, tag: Int) -> NSSlider {
            let s = NSSlider(value: value, minValue: min, maxValue: max,
                             target: self, action: #selector(sliderChanged(_:)))
            s.tag = tag
            s.widthAnchor.constraint(equalToConstant: sliderWidth).isActive = true
            return s
        }

        func separator() -> NSBox {
            let b = NSBox(); b.boxType = .separator
            b.widthAnchor.constraint(equalToConstant: sliderWidth).isActive = true
            return b
        }

        func button(_ title: String, action: Selector) -> NSButton {
            let b = NSButton(title: title, target: self, action: action)
            b.bezelStyle = .rounded
            b.widthAnchor.constraint(equalToConstant: sliderWidth).isActive = true
            return b
        }

        stack.addArrangedSubview(heading("IMAGE"))
        stack.addArrangedSubview(button("Open Image…", action: #selector(openImage)))
        stack.addArrangedSubview(NSView()) // small spacer
        stack.addArrangedSubview(separator())

        stack.addArrangedSubview(heading("CONTRAST"))
        stack.addArrangedSubview(slider(min: 0.1, max: 3.0, value: 1.0, tag: 0))

        stack.addArrangedSubview(heading("BOX BLUR RADIUS"))
        stack.addArrangedSubview(slider(min: 0, max: 15, value: 0, tag: 1))

        stack.addArrangedSubview(heading("UNSHARP STRENGTH"))
        stack.addArrangedSubview(slider(min: 0, max: 3, value: 0, tag: 2))

        stack.addArrangedSubview(separator())

        // Flexible spacer pushes export button to bottom
        let spacer = NSView()
        spacer.setContentHuggingPriority(.defaultLow, for: .vertical)
        stack.addArrangedSubview(spacer)

        stack.addArrangedSubview(button("Export PNG…", action: #selector(exportPNG)))

        panel.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.topAnchor.constraint(equalTo: panel.topAnchor),
            stack.leadingAnchor.constraint(equalTo: panel.leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: panel.trailingAnchor),
            stack.bottomAnchor.constraint(equalTo: panel.bottomAnchor),
        ])

        return panel
    }

    @objc private func sliderChanged(_ sender: NSSlider) {
        guard let p = processor else { return }
        switch sender.tag {
        case 0: p.settings.contrast        = Float(sender.doubleValue)
        case 1: p.settings.blurRadius      = Int32(sender.doubleValue)
        case 2: p.settings.unsharpStrength = Float(sender.doubleValue)
        default: break
        }
        p.process()
    }

    @objc private func openImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .tiff, .heic, .bmp]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        load(url: url)
    }

    @objc private func exportPNG() {
        guard processor?.hasImage == true else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes    = [.png]
        panel.nameFieldStringValue   = "output.png"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        processor?.export(to: url)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

// MARK: - Entry point

@main
enum ImageProcessorApp {
    @MainActor static func main() {
        let app      = NSApplication.shared
        let delegate = AppDelegate()
        app.setActivationPolicy(.regular)
        app.delegate = delegate
        app.run()
    }
}
