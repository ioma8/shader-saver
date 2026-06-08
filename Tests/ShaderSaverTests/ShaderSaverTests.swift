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
    #expect(ShaderSource.metal.contains("float2 uv = (fragCoord - r * 0.5) / r.y;"))
}

@Test func shaderFadesTowardScreenEdges() {
    #expect(ShaderSource.metal.contains("o.rgb *= 0.88 + 0.12 * exp(-length(uv) * 1.4);"))
}

@Test func shaderBreaksTheSphericalDistanceField() {
    #expect(ShaderSource.metal.contains("float core = smoothstep(1.55, 0.15, length(q * float3(0.78, 1.02, 0.78)));"))
    #expect(ShaderSource.metal.contains("float haze = smoothstep(2.1, 0.35, length(q * float3(0.58, 0.86, 0.58)));"))
}

@Test func shaderRunsAtHalfOriginalTimeSpeed() {
    #expect(ShaderSource.metal.contains("float t = uniforms.time / 10.0;"))
    #expect(ShaderSource.metal.contains("float flow = uniforms.time * 0.22;"))
}

@Test func shaderUsesRawCosineMatrixInsteadOfRotationRewrite() {
    #expect(ShaderSource.metal.contains("float2x2 rotation = float2x2(cos(t), -sin(t), sin(t), cos(t));"))
    #expect(ShaderSource.metal.contains("ro.xz = rotation * ro.xz;"))
}

@Test func shaderAddsSlightlyStrongerTimeDrivenTurbulence() {
    #expect(ShaderSource.metal.contains("q += 0.35 * sin(q.yzx * 1.4 + flow);"))
    #expect(ShaderSource.metal.contains("q += 0.22 * sin(q.zxy * 2.1 - flow * 0.8);"))
    #expect(ShaderSource.metal.contains("q += 0.12 * sin(q.xyz * 4.0 + float3(flow * 1.1, -flow * 0.9, flow * 0.7));"))
}

@Test func shaderBlendsMultipleLiquidColorBands() {
    #expect(ShaderSource.metal.contains("float swirlA = 0.5 + 0.5 * sin(q.x * 1.5 + q.y * 0.9 + flow * 0.8);"))
    #expect(ShaderSource.metal.contains("float swirlB = 0.5 + 0.5 * sin(q.z * 1.9 - q.x * 0.7 - flow * 0.6);"))
    #expect(ShaderSource.metal.contains("float3 colorA = float3(1.02, 0.34, 0.14);"))
    #expect(ShaderSource.metal.contains("float3 colorB = float3(0.92, 0.18, 0.30);"))
    #expect(ShaderSource.metal.contains("float3 colorC = float3(0.98, 0.72, 0.22);"))
}

@Test func shaderSourceCompilesIntoMetalLibrary() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    #expect(throws: Never.self) {
        _ = try device.makeLibrary(source: ShaderSource.metal, options: nil)
    }
}
