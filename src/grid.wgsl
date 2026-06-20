// CoEngine — Milestone 1, Step 1: grid shader
//
// A shader is a small program that runs on the GPU. This one has two stages:
//  - the vertex stage runs once per vertex and decides where it lands on screen
//  - the fragment stage runs once per pixel and decides its color
//
// WGSL is the shading language wgpu uses.

// The camera's combined view+projection matrix, uploaded from Rust each frame.
struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// What each vertex carries (matches the Rust `Vertex` struct).
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

// What the vertex stage passes on to the fragment stage.
struct VertexOutput {
    // `@builtin(position)` is the required final on-screen position.
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Transform the 3D world position into clip space using the camera.
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Just output the line's color, fully opaque.
    return vec4<f32>(in.color, 1.0);
}
