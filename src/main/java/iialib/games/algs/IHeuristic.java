package iialib.games.algs;

import iialib.games.model.IBoard;
import iialib.games.model.IRole;

@FunctionalInterface
public interface IHeuristic<Board extends IBoard<?, Role, Board>, Role extends IRole> {

    public static int MIN_VALUE = java.lang.Integer.MIN_VALUE;
    public static int MAX_VALUE = java.lang.Integer.MAX_VALUE;

    int eval(Board board, Role role);

}
 