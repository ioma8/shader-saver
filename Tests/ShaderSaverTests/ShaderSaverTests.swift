import Testing
import CoreGraphics
import Metal
@testable import ShaderSaver

@Test func scaledDrawableSizeReducesRetinaWorkload() {
    let size = scaledDrawableSize(bounds: CGRect(x: 0, y: 0, width: 1512, height: 982), scaleFactor: 2.0)
    #expect(abs(size.width - 1814.4) < 0.001)
    #expect(abs(size.height - 1178.4) < 0.001)
}

@Test func shaderPreservesShadertoyCoordinateSemantics() {
    #expect(ShaderSource.metal.contains("float4 FC = float4(fragCoord, 0.5, 1.0);"))
}

@Test func shaderRunsAtHalfOriginalTimeSpeed() {
    #expect(ShaderSource.metal.contains("float t = uniforms.time * 0.5;"))
}

@Test func shaderUsesRawCosineMatrixInsteadOfRotationRewrite() {
    #expect(ShaderSource.metal.contains("float4 m = cos(p.y + t + float4(0.0, 11.0, 33.0, 0.0));"))
    #expect(ShaderSource.metal.contains("float2 zx = p.zx * float2x2(m.x, m.y, m.z, m.w);"))
}

@Test func shaderAddsSlightlyStrongerTimeDrivenTurbulence() {
    #expect(ShaderSource.metal.contains("p += 1.08 * sin(p.yzx * d + t * 0.2) / d;"))
}

@Test func shaderSourceCompilesIntoMetalLibrary() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    #expect(throws: Never.self) {
        _ = try device.makeLibrary(source: ShaderSource.metal, options: nil)
    }
}
