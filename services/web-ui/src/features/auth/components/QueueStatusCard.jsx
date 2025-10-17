import { Activity, Loader2, Play, X } from 'lucide-react';
import StatusMetric from './StatusMetric';

const QueueStatusCard = ({
  queueData,
  isLoading,
  onRefresh,
  interactiveForm,
  onInteractiveChange,
  onInteractiveSubmit,
  isSubmitting,
  activeTaskId,
  task,
}) => (
  <section className="bg-white rounded-lg shadow p-6 space-y-4">
    <header className="flex items-center justify-between">
      <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
        <Activity className="h-5 w-5 text-indigo-500" /> Queue Status
      </h2>
      <button
        onClick={onRefresh}
        className="text-sm text-indigo-600 hover:text-indigo-800"
      >
        Refresh
      </button>
    </header>

    <div className="grid grid-cols-2 gap-3 text-sm">
      <StatusMetric label="Active" value={queueData?.active_tasks ?? 0} />
      <StatusMetric label="Completed" value={queueData?.completed_tasks ?? 0} />
      <StatusMetric label="In Queue" value={queueData?.queue_size ?? 0} />
      <StatusMetric label="Workers" value={queueData?.max_workers ?? 0} />
    </div>

    {isLoading && <div className="text-sm text-gray-500">Loading queue metrics…</div>}

    <div className="border-t border-gray-100 pt-4">
      <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
        <Play className="h-4 w-4" /> Trigger Interactive Login
      </h3>
      <form
        className="space-y-3"
        onSubmit={(event) => {
          event.preventDefault();
          onInteractiveSubmit();
        }}
      >
        <fieldset className="space-y-3">
          <LabeledInput
            label="Domain"
            required
            value={interactiveForm.domain}
            onChange={(value) => onInteractiveChange('domain', value)}
          />
          <LabeledInput
            label="Login URL"
            required
            type="url"
            value={interactiveForm.loginUrl}
            onChange={(value) => onInteractiveChange('loginUrl', value)}
          />
          <div className="grid grid-cols-2 gap-3">
            <LabeledSelect
              label="Auth Method"
              value={interactiveForm.authMethod}
              onChange={(value) => onInteractiveChange('authMethod', value)}
              options={[
                { value: 'form', label: 'Form' },
                { value: 'oauth', label: 'OAuth' },
              ]}
            />
            <LabeledSelect
              label="Browser"
              value={interactiveForm.browserType}
              onChange={(value) => onInteractiveChange('browserType', value)}
              options={[
                { value: 'playwright', label: 'Playwright' },
                { value: 'chrome', label: 'Chrome (Selenium)' },
                { value: 'firefox', label: 'Firefox (Selenium)' },
              ]}
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <LabeledInput
              label="Username"
              value={interactiveForm.username}
              onChange={(value) => onInteractiveChange('username', value)}
            />
            <LabeledInput
              label="Password"
              type="password"
              value={interactiveForm.password}
              onChange={(value) => onInteractiveChange('password', value)}
            />
          </div>
          <div className="flex items-center justify-between text-xs text-gray-600">
            <label className="inline-flex items-center">
              <input
                type="checkbox"
                checked={interactiveForm.headless}
                onChange={(event) => onInteractiveChange('headless', event.target.checked)}
                className="h-4 w-4 text-indigo-600 border-gray-300 rounded"
              />
              <span className="ml-2">Headless mode</span>
            </label>
            <button
              type="submit"
              disabled={isSubmitting}
              className="inline-flex items-center px-3 py-1.5 rounded-md text-white bg-indigo-600 hover:bg-indigo-700 text-xs font-medium disabled:opacity-50"
            >
              {isSubmitting ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <Play className="h-4 w-4 mr-2" />
              )}
              Queue Task
            </button>
          </div>
        </fieldset>
      </form>
    </div>

    {activeTaskId && (
      <div className="border-t border-gray-100 pt-4 text-xs space-y-1">
        <div className="flex items-center justify-between">
          <span className="font-semibold text-gray-700">Task #{activeTaskId.slice(0, 8)}</span>
          <span className="uppercase tracking-wide text-gray-500">{task?.status}</span>
        </div>
        {task?.error_message && (
          <p className="text-red-600 flex items-center gap-1"><X className="h-3 w-3" /> {task.error_message}</p>
        )}
        {task?.session_id && (
          <p className="text-green-600">Session created: {task.session_id}</p>
        )}
      </div>
    )}
  </section>
);

const LabeledInput = ({ label, type = 'text', required = false, value, onChange }) => (
  <div>
    <label className="block text-xs font-medium text-gray-600 mb-1">
      {label}
      {required ? <span className="ml-1 text-red-500">*</span> : null}
    </label>
    <input
      type={type}
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className="w-full rounded-md border-gray-300 focus:border-indigo-500 focus:ring-indigo-500"
      required={required}
    />
  </div>
);

const LabeledSelect = ({ label, value, onChange, options }) => (
  <div>
    <label className="block text-xs font-medium text-gray-600 mb-1">{label}</label>
    <select
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className="w-full rounded-md border-gray-300 focus:border-indigo-500 focus:ring-indigo-500"
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  </div>
);

export default QueueStatusCard;
