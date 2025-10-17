import { BookOpen, Database, FileText } from 'lucide-react';

export const formatIcons = {
  markdown: FileText,
  notion: Database,
  obsidian: BookOpen,
  word: FileText,
  json: Database,
  csv: Database,
};

export const formatDescriptions = {
  markdown: 'Standard Markdown files with frontmatter metadata',
  notion: 'Structured database pages with proper formatting',
  obsidian: 'Markdown files optimized for Obsidian with internal linking',
  word: 'Microsoft Word documents with tables and formatting',
  json: 'Structured JSON data for programmatic access',
  csv: 'Comma-separated values for spreadsheet applications',
};
