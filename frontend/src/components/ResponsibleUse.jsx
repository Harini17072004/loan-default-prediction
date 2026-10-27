import { InfoIcon, ShieldCheckIcon, UserIcon } from "./Icons";

const PRACTICES = [
  {
    title: "Decision support, not automation",
    body: "A probability is not a credit decision. Real deployments need policy rules, appeals, and a human reviewer.",
  },
  {
    title: "Check for bias and drift",
    body: "Historical lending data can encode unfair patterns. Monitor subgroup performance and recalibrate when the population changes.",
  },
  {
    title: "Protect applicant data",
    body: "This demo sends data only to a local API. Production systems need consent, access controls, retention limits, and audit logs.",
  },
];

export default function ResponsibleUse() {
  return (
    <section className="responsible" aria-labelledby="responsible-title">
      <div className="responsible__intro">
        <span className="section-heading__kicker">
          <ShieldCheckIcon size={16} />
          Responsible use
        </span>
        <h2 id="responsible-title">A portfolio model, not a lending policy</h2>
        <p>
          The model is trained on historical data for an educational project.
          It can help explain risk patterns, but it cannot determine whether a
          real person should receive credit.
        </p>
        <p className="responsible__callout">
          <InfoIcon size={17} />
          Do not use this dashboard for real lending, pricing, or eligibility
          decisions.
        </p>
      </div>

      <div className="responsible__grid">
        {PRACTICES.map((practice) => (
          <article className="practice" key={practice.title}>
            <UserIcon size={19} />
            <h3>{practice.title}</h3>
            <p>{practice.body}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
