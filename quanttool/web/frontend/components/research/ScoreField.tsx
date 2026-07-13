'use client';

interface ScoreFieldProps {
  name: string;
  label: string;
  description: string;
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}

export default function ScoreField({
  name,
  label,
  description,
  value,
  onChange,
  disabled = false,
}: ScoreFieldProps) {
  const descriptionId = `${name}-description`;

  return (
    <div className="grid min-h-[108px] gap-3 border-b border-border-primary py-4 sm:grid-cols-[minmax(0,1fr)_minmax(180px,240px)] sm:items-center">
      <div className="min-w-0">
        <div className="flex items-center justify-between gap-3">
          <label htmlFor={name} className="text-sm font-medium text-text-primary">
            {label}
          </label>
          <output
            htmlFor={name}
            className="min-w-[58px] text-right text-sm font-semibold tabular-nums text-primary sm:hidden"
          >
            {value.toFixed(1)} / 5
          </output>
        </div>
        <p id={descriptionId} className="mt-1 text-xs leading-5 text-text-muted">
          {description}
        </p>
      </div>
      <div className="flex h-10 items-center gap-3">
        <input
          id={name}
          name={name}
          type="range"
          min={0}
          max={5}
          step={0.5}
          value={value}
          disabled={disabled}
          aria-describedby={descriptionId}
          onChange={(event) => onChange(Number(event.target.value))}
          className="h-2 w-full cursor-pointer accent-primary disabled:cursor-not-allowed disabled:opacity-50"
        />
        <output
          htmlFor={name}
          className="hidden min-w-[58px] text-right text-sm font-semibold tabular-nums text-primary sm:block"
        >
          {value.toFixed(1)} / 5
        </output>
      </div>
    </div>
  );
}
