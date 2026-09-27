/**
 * Section navigation.
 *
 * The wordmark, its expansion and the Refresh / Raw Data / Presentation
 * controls belong to the shared dashboard header, so this deliberately does not
 * repeat them. It contributes the one thing that header cannot: a rule of
 * section tabs, with the active one marked by a cyan underline and
 * `aria-current` so it is announced as well as shown.
 */

import React from 'react';
import { SECTIONS, PRESENTATION_SECTIONS } from '../../phase2/pipeline.js';

export function Masthead({ active, onSelect, presentationMode }) {
  const sections = presentationMode ? SECTIONS.filter((s) => PRESENTATION_SECTIONS.has(s.id)) : SECTIONS;

  return (
    <>
      <nav className="rc-nav" aria-label="Research sections">
        {sections.map((section) => (
          <button
            key={section.id}
            type="button"
            className="rc-nav-btn"
            aria-current={active === section.id}
            onClick={() => onSelect(section.id)}
          >
            {section.label}
          </button>
        ))}
      </nav>
    </>
  );
}
