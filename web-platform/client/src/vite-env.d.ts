// Type declarations for YAML imports in Vite
declare module '*.yaml' {
  const content: any;
  export default content;
}

declare module '*.yml' {
  const content: any;
  export default content;
}
