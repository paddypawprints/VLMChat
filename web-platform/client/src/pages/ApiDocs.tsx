export default function ApiDocs() {
  return (
    <div className="h-screen w-full">
      <iframe
        src="/docs/openapi/index.html"
        className="w-full h-full border-0"
        title="API Documentation"
      />
    </div>
  );
}
