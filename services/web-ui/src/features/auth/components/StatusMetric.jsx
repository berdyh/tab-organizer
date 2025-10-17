const StatusMetric = ({ label, value }) => (
  <div className="bg-gray-50 rounded-md p-3">
    <div className="text-xs text-gray-500 uppercase">{label}</div>
    <div className="text-xl font-semibold text-gray-900">{value}</div>
  </div>
);

export default StatusMetric;
