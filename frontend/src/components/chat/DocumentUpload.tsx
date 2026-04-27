"use client";

import { useState } from "react";
import { UploadCloud, File, CheckCircle, AlertCircle } from "lucide-react";
import { apiUrl } from "@/lib/api";

export default function DocumentUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<"idle" | "uploading" | "success" | "error">("idle");
  const [message, setMessage] = useState("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setStatus("idle");
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setStatus("uploading");
    const formData = new FormData();
    formData.append("file", file);
    // Optional namespace
    formData.append("namespace", "default");

    try {
      const response = await fetch(apiUrl("/ingest"), {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const data = await response.json();
      setStatus("success");
      setMessage(`Successfully ingested ${data.chunks} chunks from ${file.name}.`);
      setFile(null);
    } catch (err: unknown) {
      console.error(err);
      setStatus("error");
      if (err instanceof Error) {
        setMessage(err.message || "An error occurred during upload.");
      } else {
        setMessage("An error occurred during upload.");
      }
    }
  };

  return (
    <div className="bg-white p-6 rounded-xl border border-gray-100 shadow-sm mt-6">
      <h2 className="text-xl font-semibold mb-4 text-gray-800">Knowledge Base</h2>
      <p className="text-sm text-gray-500 mb-6">Upload PDF documents to ground the agent&apos;s answers in your data.</p>
      
      <div className="border-2 border-dashed border-gray-200 rounded-lg p-6 text-center hover:bg-gray-50 transition-colors">
        <input
          type="file"
          id="file-upload"
          className="hidden"
          accept=".pdf"
          onChange={handleFileChange}
        />
        <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center">
          <UploadCloud className="w-10 h-10 text-indigo-500 mb-2" />
          <span className="text-sm font-medium text-gray-700">Click to select a PDF file</span>
          <span className="text-xs text-gray-400 mt-1">Maximum 10MB</span>
        </label>
      </div>

      {file && (
        <div className="mt-4 flex items-center justify-between bg-gray-50 p-3 rounded-lg border border-gray-100">
          <div className="flex items-center">
            <File className="w-5 h-5 text-gray-500 mr-3" />
            <span className="text-sm font-medium text-gray-700 truncate w-48">{file.name}</span>
          </div>
          <button
            onClick={handleUpload}
            disabled={status === "uploading"}
            className="px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-md hover:bg-indigo-700 disabled:opacity-50 transition-colors"
          >
            {status === "uploading" ? "Uploading..." : "Upload & Ingest"}
          </button>
        </div>
      )}

      {status === "success" && (
        <div className="mt-4 flex items-start text-green-700 bg-green-50 p-3 rounded-lg border border-green-100">
          <CheckCircle className="w-5 h-5 mr-2 shrink-0 mt-0.5" />
          <span className="text-sm">{message}</span>
        </div>
      )}

      {status === "error" && (
        <div className="mt-4 flex items-start text-red-700 bg-red-50 p-3 rounded-lg border border-red-100">
          <AlertCircle className="w-5 h-5 mr-2 shrink-0 mt-0.5" />
          <span className="text-sm">{message}</span>
        </div>
      )}
    </div>
  );
}
