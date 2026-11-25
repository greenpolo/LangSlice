// What the viewer shows: the agent's sections until the person picks one.
export type Focus = {
  ids: string[];               // the sections to show
  following: boolean;          // the agent's choice, else the person's
  pick: (id: string) => void;
  follow: () => void;
};
