import { NotConfigured } from "../components/ui";

export default function Experiments() {
  return (
    <>
      <h1>Experiments</h1>
      <NotConfigured what="no experiment results" detail="Baselines, cross-play and confidence intervals appear once the evaluation pipeline writes results." />
    </>
  );
}
