import pandas as pd

import numpy as np
df = pd.read_csv('../input/train.csv')
not_round = ['toxicity_annotator_count', 'identity_annotator_count', 'created_date',

                 'publication_id', 'parent_id', 'article_id', 'rating', 'funny',

                 'wow', 'sad', 'likes', 'disagree', 'rating', 'comment_text', 'target',

                'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit',

                'id', 'number_of_records']

round_cols = list(set(df.columns) - set(not_round))
df_round = df

for i in round_cols:

    df_round[i] = df_round[i].round(1)
df_round['is_gender'] = False

df_round['is_race'] = False

df_round['is_religion'] = False

df_round['is_sexual_orientation'] = False

df_round['is_disability'] = False
df_round.loc[((df['female'].notnull()) & (df['female'] > 0)) |

((df['male'].notnull()) & (df['male'] > 0)) |

((df['other_gender'].notnull()) & (df['other_gender'] > 0)), 'is_gender'] = True
df_round.loc[((df_round['asian'].notnull()) & (df_round['asian'] > 0)) |

((df_round['black'].notnull()) & (df_round['black'] > 0)) |

((df_round['other_race_or_ethnicity'].notnull()) & (df_round['other_race_or_ethnicity'] > 0)) |

((df_round['white'].notnull()) & (df_round['white'] > 0)) |

((df_round['latino'].notnull()) & (df_round['latino'] > 0)), 'is_race'] = True
df_round.loc[((df_round['atheist'].notnull()) & (df_round['atheist'] > 0)) |

((df_round['buddhist'].notnull()) & (df_round['buddhist'] > 0)) |

((df_round['christian'].notnull()) & (df_round['christian'] > 0)) |

((df_round['hindu'].notnull()) & (df_round['hindu'] > 0)) |

((df_round['jewish'].notnull()) & (df_round['jewish'] > 0)) |

((df_round['muslim'].notnull()) & (df_round['muslim'] > 0)) |

((df_round['other_religion'].notnull()) & (df_round['other_religion'] > 0)), 'is_religion'] = True
df_round.loc[((df_round['bisexual'].notnull()) & (df_round['bisexual'] > 0)) |

((df_round['heterosexual'].notnull()) & (df_round['heterosexual'] > 0)) |

((df_round['homosexual_gay_or_lesbian'].notnull()) & (df_round['homosexual_gay_or_lesbian'] > 0)) |

((df_round['other_sexual_orientation'].notnull()) & (df_round['other_sexual_orientation'] > 0)) |

((df_round['transgender'].notnull()) & (df_round['transgender'] > 0)), 'is_sexual_orientation'] = True
df_round.loc[((df_round['intellectual_or_learning_disability'].notnull()) & (df_round['intellectual_or_learning_disability'] > 0)) |

((df_round['other_disability'].notnull()) & (df_round['other_disability'] > 0)) |

((df_round['physical_disability'].notnull()) & (df_round['physical_disability'] > 0)) |

((df_round['psychiatric_or_mental_illness'].notnull()) & (df_round['psychiatric_or_mental_illness'] > 0)), 'is_disability'] = True
df_round.to_csv('jigsaw_new_categories.csv', encoding='utf-8', index=False, sep='\t')