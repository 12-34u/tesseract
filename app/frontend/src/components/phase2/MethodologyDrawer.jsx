/**
 * Implementation reference.
 *
 * Architecture, training protocol, seed policy, amendment status and
 * provenance. Kept collapsed and last, and hidden entirely in presentation
 * mode, so the research sections above stay about the research.
 */

import React from 'react';
import { Disclosure, KeyValues, RuleHeading } from './primitives.jsx';
import { methodologyGroups, sourcePaths } from '../../phase2/pipeline.js';

export function MethodologyDrawer({ status, showSources = true }) {
  const groups = methodologyGroups(status);
  const paths = showSources ? sourcePaths(status) : [];
  if (!groups.length && !paths.length) return null;

  return (
    <Disclosure label="Implementation reference" hint="architecture, protocol, seeds, git state">
      <div className="rc-grid rc-cols-2">
        {groups.map((group) => (
          <div key={group.id}>
            <RuleHeading>{group.title}</RuleHeading>
            <KeyValues fields={group.fields} />
            {group.note ? <p className="rc-consequence" style={{ marginTop: '0.6rem' }}>{group.note}</p> : null}
          </div>
        ))}
      </div>

      {paths.length > 0 && (
        <div style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid var(--rc-line)' }}>
          <RuleHeading>Artifacts behind this view</RuleHeading>
          <ul className="rc-list">
            {paths.map((path) => (
              <li className="rc-source" key={path}>{path}</li>
            ))}
          </ul>
        </div>
      )}
    </Disclosure>
  );
}
