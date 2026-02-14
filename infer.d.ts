export interface NativeModule {
    vecmatmul(out: Float32Array, M: Float32Array, x: Float32Array): any;
}

declare module "*.node" {
    const mod: NativeModule;
    export default mod;
}
