"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { citedNumbers, linkifyCitations, type CitationIndex } from "@/lib/citations";
import { cn } from "@/lib/utils";
import { CitationChip } from "./CitationChip";

/** Answer text: markdown, with `[n]` markers rendered as citation chips. */
export function Markdown({ text, index, className }: { text: string; index: CitationIndex; className?: string }) {
  return (
    <div className={cn("prose-mono", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a({ href, children }) {
            const nums = citedNumbers(href);
            if (nums) {
              return (
                <span className="whitespace-nowrap">
                  {nums.map((n) => (
                    <CitationChip key={n} n={n} finding={index.finding(n)} />
                  ))}
                </span>
              );
            }
            return (
              <a href={href} target="_blank" rel="noreferrer noopener">
                {children}
              </a>
            );
          },
          h1: ({ children }) => <p className="font-semibold">{children}</p>,
          h2: ({ children }) => <p className="font-semibold">{children}</p>,
          h3: ({ children }) => <p className="font-semibold">{children}</p>,
        }}
      >
        {linkifyCitations(text)}
      </ReactMarkdown>
    </div>
  );
}
