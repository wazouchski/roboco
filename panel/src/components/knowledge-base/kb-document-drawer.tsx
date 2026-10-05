"use client";

import { FileText, Hash } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { useKBDocument } from "@/hooks/use-knowledge-base";
import type { KBSearchResult } from "@/types";
import { KBIndexTypeBadge } from "./kb-index-type-badge";

interface KBDocumentDrawerProps {
  result: KBSearchResult | null;
  onOpenChange: (open: boolean) => void;
}

export function KBDocumentDrawer({
  result,
  onOpenChange,
}: KBDocumentDrawerProps) {
  const document = useKBDocument(
    result?.index_type ?? null,
    result?.source ?? null,
  );

  return (
    <Sheet open={Boolean(result)} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full sm:max-w-3xl">
        <SheetHeader>
          <div className="flex items-center gap-2 pr-8">
            {result && <KBIndexTypeBadge indexType={result.index_type} />}
            <FileText className="h-4 w-4 text-muted-foreground" />
          </div>
          <SheetTitle className="break-all pr-8 text-left font-mono text-sm">
            {result?.source ?? "Knowledge base document"}
          </SheetTitle>
          <SheetDescription>
            {document.isLoading
              ? "Loading the indexed document…"
              : document.data
                ? `${document.data.chunk_count} indexed chunk${document.data.chunk_count === 1 ? "" : "s"}`
                : "Full indexed source"}
          </SheetDescription>
        </SheetHeader>

        <ScrollArea className="min-h-0 flex-1 pr-4">
          {document.isError ? (
            <p className="text-sm text-destructive">
              Unable to load this document.
            </p>
          ) : document.isLoading ? (
            <p className="text-sm text-muted-foreground">Loading…</p>
          ) : (
            <>
              {document.data?.metadata &&
                Object.keys(document.data.metadata).length > 0 && (
                  <div className="mb-4 flex flex-wrap gap-2">
                    <Badge variant="secondary">
                      <Hash className="mr-1 h-3 w-3" />
                      {document.data.chunk_count} chunks
                    </Badge>
                  </div>
                )}
              <pre className="whitespace-pre-wrap break-words rounded-md bg-muted/40 p-4 font-mono text-xs leading-5">
                {document.data?.content ?? "No indexed content found."}
              </pre>
            </>
          )}
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
