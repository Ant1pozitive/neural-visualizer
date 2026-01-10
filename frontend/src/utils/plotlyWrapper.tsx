// src/utils/plotlyWrapper.tsx
import React from 'react';
import Plot from 'react-plotly.js';

/**
 * Light wrapper around react-plotly.js to keep consistent props
 * across charts. This module centralizes configuration (responsive,
 * layout defaults) so rest of components can be concise.
 */

/** Default layout that works well with Tailwind panels */
const defaultLayout: Partial<Plotly.Layout> = {
  autosize: true,
  margin: { l: 50, r: 20, t: 40, b: 50 },
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)'
};

type PlotlyWrapperProps = React.ComponentProps<typeof Plot> & {
  className?: string;
};

export default function PlotlyWrapper(props: PlotlyWrapperProps) {
  const { data, layout, useResizeHandler = true, style = { width: '100%', height: '100%' }, ...rest } = props;
  return (
    <Plot
      data={data}
      layout={{ ...defaultLayout, ...(layout || {}) }}
      useResizeHandler={useResizeHandler}
      style={style}
      {...rest}
    />
  );
}
