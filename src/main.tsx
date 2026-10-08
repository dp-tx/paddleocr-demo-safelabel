import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import LabelLens from "../app/page";
import "../app/globals.css";

const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error("The page is missing its #root element.");
}

createRoot(rootElement).render(
  <StrictMode>
    <LabelLens />
  </StrictMode>,
);
