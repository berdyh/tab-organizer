import { Loader2, KeyRound } from 'lucide-react';

const CredentialForm = ({ form, onChange, onSubmit, isSubmitting }) => (
  <section className="xl:col-span-2 bg-white rounded-lg shadow p-6">
    <h2 className="text-lg font-semibold text-gray-900 mb-4">Store Domain Credentials</h2>
    <form
      className="grid grid-cols-1 md:grid-cols-2 gap-4"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      <Field label="Domain" required>
        <input
          type="text"
          value={form.domain}
          onChange={(event) => onChange('domain', event.target.value)}
          placeholder="example.com"
          className="w-full rounded-md border-gray-300 focus:border-indigo-500 focus:ring-indigo-500"
          required
        />
      </Field>
      <Field label="Login URL" required>
        <input
          type="url"
          value={form.loginUrl}
          onChange={(event) => onChange('loginUrl', event.target.value)}
          placeholder="https://example.com/login"
          className="w-full rounded-md border-gray-300 focus:border-indigo-500 focus:ring-indigo-500"
          required
        />
      </Field>
      <Field label="Auth Method">
        <select
          value={form.authMethod}
          onChange={(event) => onChange('authMethod', event.target.value)}
          className="w-full rounded-md border-gray-300 focus:border-indigo-500 focus:ring-indigo-500"
        >
          <option value="form">Form-Based Login</option>
          <option value="oauth">OAuth</option>
          <option value="basic">Basic Auth</option>
        </select>
      </Field>
      <Field label="Username / Email">
        <input
          type="text"
          value={form.username}
          onChange={(event) => onChange('username', event.target.value)}
          placeholder="user@example.com"
          className="w-full rounded-md border-gray-300 focus:border-indigo-500 focus:ring-indigo-500"
        />
      </Field>
      <Field label="Password / Secret">
        <input
          type="password"
          value={form.password}
          onChange={(event) => onChange('password', event.target.value)}
          className="w-full rounded-md border-gray-300 focus:border-indigo-500 focus:ring-indigo-500"
        />
      </Field>
      <div className="md:col-span-2 flex justify-end">
        <button
          type="submit"
          disabled={isSubmitting}
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50"
        >
          {isSubmitting ? (
            <Loader2 className="h-4 w-4 animate-spin mr-2" />
          ) : (
            <KeyRound className="h-4 w-4 mr-2" />
          )}
          Save Credentials
        </button>
      </div>
    </form>
  </section>
);

const Field = ({ label, required = false, children }) => (
  <div>
    <label className="block text-sm font-medium text-gray-700 mb-1">
      {label}
      {required ? <span className="ml-1 text-red-500">*</span> : null}
    </label>
    {children}
  </div>
);

export default CredentialForm;
